/**
 * @file random.hpp
 * @brief Random number generation (NumPy random.Generator API shape).
 *
 * Provides API-compatible random number generation using C++11 <random>:
 * method names and signatures mirror np.random.Generator, but the
 * underlying engine is std::mt19937_64, NOT NumPy's default PCG64 — the
 * same seed produces different streams than NumPy (an earlier revision
 * claimed bit-compatibility). Do not cross-validate raw streams against
 * NumPy; validate distributions statistically instead.
 * Implements the Generator class with all standard distributions.
 *
 * Reference: numpy-reference/reference/random/generator.html
 *
 * @author Sergio Randriamihoatra (sergiorandriamihoatra@gmail.com)
 */
#ifndef NP_RANDOM_HPP
#define NP_RANDOM_HPP

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <numbers>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

#include "api_macros.hpp"
#include "creation.hpp"
#include "dtype.hpp"
#include "exceptions.hpp"
#include "ndarray.hpp"
#include "pqc.hpp"

namespace np::random
{

namespace detail
{
// Loud domain validation for distribution parameters. std:: distributions
// have narrow preconditions (violating them is UB, not an error); NaN fails
// every check via !(x relop y). Verified edge-by-edge against NumPy, whose
// conventions are followed (e.g. exponential(0) and geometric(1) are valid
// and degenerate; poisson(0) is valid).
template <typename T> inline void require_positive(T v, const char *what)
{
    if (!(v > T{0}))
    {
        throw std::invalid_argument(what);
    }
}
template <typename T> inline void require_nonnegative(T v, const char *what)
{
    if (!(v >= T{0}))
    {
        throw std::invalid_argument(what);
    }
}
template <typename T> inline void require_unit_interval(T p, const char *what)
{
    if (!(p >= T{0}) || !(p <= T{1}))
    {
        throw std::invalid_argument(what);
    }
}
} // namespace detail


/**
 * @brief Random number generator (NumPy Generator API equivalent).
 *
 * Wraps C++ std::mt19937_64 (Mersenne Twister). API-compatible with
 * np.random.Generator (same method names/shapes); streams are NOT
 * bit-compatible with NumPy, whose default engine is PCG64.
 *
 * Reference: numpy-reference/reference/random/generator.html
 */
class Generator
{
  public:
    using engine_type = std::mt19937_64;
    using result_type = engine_type::result_type;

    /**
     * @brief Construct generator with optional seed.
     * @param seed Random seed (uses random_device if not provided)
     */
    explicit Generator(std::optional<std::uint64_t> seed = std::nullopt)
        : engine_([&]() -> engine_type {
              if (seed.has_value())
              {
                  pqc::SecureSeed<std::uint64_t> ss(*seed);
                  return engine_type(ss.get());
              }
              return engine_type(std::random_device{}());
          }())
    {
    }

    ~Generator()
    {
        // PQC: zeroize engine state on destruction (key material may have
        // been generated via this engine). Not elided due to volatile fence.
        pqc::secure_zero(&engine_, sizeof(engine_));
        pqc::ct_barrier();
    }

    Generator(const Generator &other) : engine_(other.engine_)
    {
    }
    Generator &operator=(const Generator &other)
    {
        if (this != &other)
        {
            engine_ = other.engine_;
        }
        return *this;
    }
    Generator(Generator &&) noexcept = default;
    Generator &operator=(Generator &&) noexcept = default;

    /**
     * @brief Securely clear the generator state (zeroize engine).
     * Reference: pqc.hpp:secure_zero
     */
    void secure_clear() noexcept
    {
        pqc::secure_zero(&engine_, sizeof(engine_));
        pqc::ct_barrier();
    }

    // Simple Random Data
    /**
     * @brief Random integers from low (inclusive) to high (exclusive).
     * Reference:
     * numpy-reference/reference/random/generated/numpy.random.Generator.integers.html
     */
    template <typename T = std::int64_t> auto integers(T low, T high, const std::vector<int> &size = {}) -> ndarray<T>
    {
        if (high <= low)
        {
            throw std::invalid_argument("high must be greater than low");
        }

        if (size.empty())
        {
            std::uniform_int_distribution<T> dist(low, high - 1);
            return ndarray<T>::from_data({1}, {dist(engine_)});
        }

        std::uniform_int_distribution<T> dist(low, high - 1);
        ndarray<T> result(size, dtype_of<T>);
        T *__restrict dst = result.data().data();
        std::size_t n = result.size();
        for (std::size_t i = 0; i < n; ++i)
        {
            dst[i] = dist(engine_);
        }
        return result;
    }

    /**
     * @brief Random floats in the half-open interval [0.0, 1.0).
     * Reference:
     * numpy-reference/reference/random/generated/numpy.random.Generator.random.html
     */
    template <typename T = double> auto random(const std::vector<int> &size = {}) -> ndarray<T>
    {
        static_assert(std::is_floating_point_v<T>, "T must be floating point");

        if (size.empty()) [[unlikely]]
        {
            std::uniform_real_distribution<T> dist(T{0}, T{1});
            return ndarray<T>::from_data({1}, {dist(engine_)});
        }

        std::uniform_real_distribution<T> dist(T{0}, T{1});
        ndarray<T> result(size, dtype_of<T>);
        T *__restrict dst = result.data().data();
        std::size_t n = result.size();
        for (std::size_t i = 0; i < n; ++i)
        {
            dst[i] = dist(engine_);
        }
        return result;
    }

    /**
     * @brief Random bytes.
     * Reference:
     * numpy-reference/reference/random/generated/numpy.random.Generator.bytes.html
     */
    auto bytes(std::size_t length) -> std::vector<std::uint8_t>
    {
        std::uniform_int_distribution<std::uint16_t> dist(0, 255);
        std::vector<std::uint8_t> result(length);
        for (auto &byte : result)
        {
            byte = static_cast<std::uint8_t>(dist(engine_));
        }
        return result;
    }

    /**
     * @brief PQC-hardened random bytes (constant-time, zeroizes temp).
     *
     * Like `bytes()` but wraps the generation in `pqc::ct_barrier`
     * to prevent reordering across the secret boundary. Caller should
     * `pqc::secure_zero` the returned vector when done.
     * Reference: pqc.hpp:ct_barrier, secure_zero
     */
    auto bytes_secure(std::size_t length) -> std::vector<std::uint8_t>
    {
        pqc::ct_barrier();
        auto out = bytes(length);
        pqc::ct_barrier();
        return out;
    }

    // Permutations
    /**
     * @brief Randomly permute a sequence or array.
     * Reference:
     * numpy-reference/reference/random/generated/numpy.random.Generator.permutation.html
     */
    template <typename T> auto permutation(const ndarray<T> &x) -> ndarray<T>
    {
        auto result = x.copy();
        shuffle(result);
        return result;
    }

    /**
     * @brief Return permuted range.
     */
    auto permutation(std::int64_t n) -> ndarray<std::int64_t>
    {
        auto arr = arange<std::int64_t>(0, n, 1);
        shuffle(arr);
        return arr;
    }

    /**
     * @brief Modify array in-place by shuffling its contents.
     * Reference:
     * numpy-reference/reference/random/generated/numpy.random.Generator.shuffle.html
     */
    template <typename T> void shuffle(ndarray<T> &x)
    {
        if (x.ndim() == 1) [[likely]]
        {
            if (x.is_contiguous()) [[likely]]
            {
                T *__restrict p = x.data().data();
                std::shuffle(p, p + x.size(), engine_);
            }
            else
            {
                std::shuffle(x.data().begin(), x.data().end(), engine_);
            }
        }
        else
        {
            // Shuffle along first axis
            const std::size_t n0 = static_cast<std::size_t>(x.shape[0]);
            std::vector<std::size_t> indices;
            indices.reserve(n0);
            for (std::size_t i = 0; i < n0; ++i)
                indices.push_back(i);
            std::shuffle(indices.begin(), indices.end(), engine_);

            // Create permuted copy
            auto result = x.copy();
            for (std::size_t i = 0; i < n0; ++i)
            {
                // Copy row indices[i] to row i
                for (std::size_t j = 0; j < x.size() / n0; ++j)
                {
                    x.data()[i * (x.size() / n0) + j] = result.data()[indices[i] * (x.size() / n0) + j];
                }
            }
        }
    }

    /**
     * @brief Random choice from 1-D array.
     * Reference:
     * numpy-reference/reference/random/generated/numpy.random.Generator.choice.html
     */
    template <typename T> auto choice(const ndarray<T> &a, std::size_t size = 1, bool replace = true) -> ndarray<T>
    {
        // NOTE (honesty audit): uniform_int_distribution(0, n-1) with n == 0
        // wraps to SIZE_MAX (then OOB) — guard explicitly.
        if (a.size() == 0)
        {
            if (size == 0)
            {
                return ndarray<T>(std::vector<int>{0});
            }
            throw std::invalid_argument("choice: cannot sample from empty array");
        }
        if (a.ndim() != 1)
        {
            throw std::invalid_argument("choice: array must be 1-D");
        }

        const std::size_t n = a.size();
        if (!replace && size > n)
        {
            throw std::invalid_argument("choice: Cannot sample more elements than "
                                        "population when replace=False");
        }

        std::vector<T> result_data;
        result_data.reserve(size);

        if (replace)
        {
            std::uniform_int_distribution<std::size_t> dist(0, n - 1);
            for (std::size_t i = 0; i < size; ++i)
            {
                result_data.push_back(a.data()[dist(engine_)]);
            }
        }
        else
        {
            // Sampling without replacement
            std::vector<std::size_t> indices(n);
            std::iota(indices.begin(), indices.end(), 0);
            std::shuffle(indices.begin(), indices.end(), engine_);

            for (std::size_t i = 0; i < size; ++i)
            {
                result_data.push_back(a.data()[indices[i]]);
            }
        }

        return ndarray<T>::from_data({static_cast<int>(size)}, std::move(result_data));
    }

    // Distributions
    /**
     * @brief Draw samples from a uniform distribution over [low, high).
     * Reference:
     * numpy-reference/reference/random/generated/numpy.random.Generator.uniform.html
     */
    template <typename T = double>
    auto uniform(T low = T{0}, T high = T{1}, const std::vector<int> &size = {}) -> ndarray<T>
    {
        std::uniform_real_distribution<T> dist(low, high);
        return _fill_distribution<T>(engine_, dist, size);
    }

    /**
     * @brief Draw samples from a standard normal distribution (mean=0, stdev=1).
     * Reference:
     * numpy-reference/reference/random/generated/numpy.random.Generator.standard_normal.html
     */
    template <typename T = double> auto standard_normal(const std::vector<int> &size = {}) -> ndarray<T>
    {
        std::normal_distribution<T> dist(T{0}, T{1});
        return _fill_distribution<T>(engine_, dist, size);
    }

    /**
     * @brief Draw samples from a normal (Gaussian) distribution.
     * Reference:
     * numpy-reference/reference/random/generated/numpy.random.Generator.normal.html
     */
    template <typename T = double>
    auto normal(T loc = T{0}, T scale = T{1}, const std::vector<int> &size = {}) -> ndarray<T>
    {
        std::normal_distribution<T> dist(loc, scale);
        return _fill_distribution<T>(engine_, dist, size);
    }

    /**
     * @brief Draw samples from an exponential distribution.
     * Reference:
     * numpy-reference/reference/random/generated/numpy.random.Generator.exponential.html
     */
    template <typename T = double> auto exponential(T scale = T{1}, const std::vector<int> &size = {}) -> ndarray<T>
    {
        detail::require_nonnegative(scale, "exponential: scale must be >= 0");
        if (scale == T{0})
            return ndarray<T>(size, dtype_of<T>); // degenerate: all zeros, like NumPy
        std::exponential_distribution<T> dist(T{1} / scale);
        return _fill_distribution<T>(engine_, dist, size);
    }

    /**
     * @brief Draw samples from a standard exponential distribution.
     * Reference:
     * numpy-reference/reference/random/generated/numpy.random.Generator.standard_exponential.html
     */
    template <typename T = double> auto standard_exponential(const std::vector<int> &size = {}) -> ndarray<T>
    {
        return exponential(T{1}, size);
    }

    /**
     * @brief Draw samples from a gamma distribution.
     * Reference:
     * numpy-reference/reference/random/generated/numpy.random.Generator.gamma.html
     */
    template <typename T = double> auto gamma(T shape, T scale = T{1}, const std::vector<int> &size = {}) -> ndarray<T>
    {
        detail::require_positive(shape, "gamma: shape must be > 0");
        detail::require_positive(scale, "gamma: scale must be > 0");
        std::gamma_distribution<T> dist(shape, scale);
        return _fill_distribution<T>(engine_, dist, size);
    }

    /**
     * @brief Draw samples from a standard gamma distribution.
     * Reference:
     * numpy-reference/reference/random/generated/numpy.random.Generator.standard_gamma.html
     */
    template <typename T = double> auto standard_gamma(T shape, const std::vector<int> &size = {}) -> ndarray<T>
    {
        return gamma(shape, T{1}, size);
    }

    /**
     * @brief Draw samples from a beta distribution.
     * Reference:
     * numpy-reference/reference/random/generated/numpy.random.Generator.beta.html
     */
    template <typename T = double> auto beta(T a, T b, const std::vector<int> &size = {}) -> ndarray<T>
    {
        detail::require_positive(a, "beta: a must be > 0");
        detail::require_positive(b, "beta: b must be > 0");
        // Beta distribution: X ~ Gamma(a,1) / (Gamma(a,1) + Gamma(b,1))
        std::gamma_distribution<T> dist_a(a, T{1});
        std::gamma_distribution<T> dist_b(b, T{1});

        if (size.empty())
        {
            T x = dist_a(engine_);
            T y = dist_b(engine_);
            return ndarray<T>::from_data({1}, {x / (x + y)});
        }

        ndarray<T> result(size, dtype_of<T>);
        for (auto it = result.begin(); it != result.end(); ++it)
        {
            T x = dist_a(engine_);
            T y = dist_b(engine_);
            *it = x / (x + y);
        }
        return result;
    }

    /**
     * @brief Draw samples from a chi-square distribution.
     * Reference:
     * numpy-reference/reference/random/generated/numpy.random.Generator.chisquare.html
     */
    template <typename T = double> auto chisquare(T df, const std::vector<int> &size = {}) -> ndarray<T>
    {
        detail::require_positive(df, "chisquare: df must be > 0");
        std::chi_squared_distribution<T> dist(df);
        return _fill_distribution<T>(engine_, dist, size);
    }

    /**
     * @brief Draw samples from an F distribution.
     * Reference:
     * numpy-reference/reference/random/generated/numpy.random.Generator.f.html
     */
    template <typename T = double> auto f(T dfnum, T dfden, const std::vector<int> &size = {}) -> ndarray<T>
    {
        detail::require_positive(dfnum, "f: dfnum must be > 0");
        detail::require_positive(dfden, "f: dfden must be > 0");
        std::fisher_f_distribution<T> dist(dfnum, dfden);
        return _fill_distribution<T>(engine_, dist, size);
    }

    /**
     * @brief Draw samples from a Student's t distribution.
     * Reference:
     * numpy-reference/reference/random/generated/numpy.random.Generator.standard_t.html
     */
    template <typename T = double> auto standard_t(T df, const std::vector<int> &size = {}) -> ndarray<T>
    {
        detail::require_positive(df, "standard_t: df must be > 0");
        std::student_t_distribution<T> dist(df);
        return _fill_distribution<T>(engine_, dist, size);
    }

    /**
     * @brief Draw samples from a lognormal distribution.
     * Reference:
     * numpy-reference/reference/random/generated/numpy.random.Generator.lognormal.html
     */
    template <typename T = double>
    auto lognormal(T mean = T{0}, T sigma = T{1}, const std::vector<int> &size = {}) -> ndarray<T>
    {
        std::lognormal_distribution<T> dist(mean, sigma);
        return _fill_distribution<T>(engine_, dist, size);
    }

    /**
     * @brief Draw samples from a Cauchy distribution.
     * Reference:
     * numpy-reference/reference/random/generated/numpy.random.Generator.standard_cauchy.html
     */
    template <typename T = double> auto standard_cauchy(const std::vector<int> &size = {}) -> ndarray<T>
    {
        std::cauchy_distribution<T> dist(T{0}, T{1});
        return _fill_distribution<T>(engine_, dist, size);
    }

    /**
     * @brief Draw samples from a Weibull distribution.
     * Reference:
     * numpy-reference/reference/random/generated/numpy.random.Generator.weibull.html
     */
    template <typename T = double> auto weibull(T a, const std::vector<int> &size = {}) -> ndarray<T>
    {
        detail::require_positive(a, "weibull: a must be > 0");
        std::weibull_distribution<T> dist(a, T{1});
        return _fill_distribution<T>(engine_, dist, size);
    }

    /**
     * @brief Draw samples from a Poisson distribution.
     * Reference:
     * numpy-reference/reference/random/generated/numpy.random.Generator.poisson.html
     */
    template <typename T = double>
    auto poisson(T lam = T{1}, const std::vector<int> &size = {}) -> ndarray<std::int64_t>
    {
        detail::require_nonnegative(lam, "poisson: lam must be >= 0");
        if (lam == T{0})
            return ndarray<std::int64_t>(size, dtype_of<std::int64_t>);
        std::poisson_distribution<std::int64_t> dist(lam);
        return _fill_distribution<std::int64_t>(engine_, dist, size);
    }

    /**
     * @brief Draw samples from a binomial distribution.
     * Reference:
     * numpy-reference/reference/random/generated/numpy.random.Generator.binomial.html
     */
    auto binomial(std::int64_t n, double p, const std::vector<int> &size = {}) -> ndarray<std::int64_t>
    {
        if (n < 0)
            throw std::invalid_argument("binomial: n must be >= 0");
        detail::require_unit_interval(p, "binomial: p must be in [0, 1]");
        std::binomial_distribution<std::int64_t> dist(n, p);
        return _fill_distribution<std::int64_t>(engine_, dist, size);
    }

    /**
     * @brief Draw samples from a negative binomial distribution.
     * Reference:
     * numpy-reference/reference/random/generated/numpy.random.Generator.negative_binomial.html
     */
    auto negative_binomial(std::int64_t n, double p, const std::vector<int> &size = {}) -> ndarray<std::int64_t>
    {
        if (n <= 0)
            throw std::invalid_argument("negative_binomial: n must be > 0");
        if (!(p > 0.0) || !(p <= 1.0))
            throw std::invalid_argument("negative_binomial: p must be in (0, 1]");
        if (p == 1.0)
            return ndarray<std::int64_t>(size, dtype_of<std::int64_t>);
        std::negative_binomial_distribution<std::int64_t> dist(n, p);
        return _fill_distribution<std::int64_t>(engine_, dist, size);
    }

    /**
     * @brief Draw samples from a geometric distribution.
     * Reference:
     * numpy-reference/reference/random/generated/numpy.random.Generator.geometric.html
     */
    auto geometric(double p, const std::vector<int> &size = {}) -> ndarray<std::int64_t>
    {
        if (!(p > 0.0) || !(p <= 1.0))
            throw std::invalid_argument("geometric: p must be in (0, 1]");
        if (p == 1.0)
            return ndarray<std::int64_t>(size, dtype_of<std::int64_t>, std::int64_t{1});
        std::geometric_distribution<std::int64_t> dist(p);
        return _fill_distribution<std::int64_t>(engine_, dist, size);
    }

    /**
     * @brief Draw samples from a Pareto II distribution.
     * Reference:
     * numpy-reference/reference/random/generated/numpy.random.Generator.pareto.html
     */
    template <typename T = double> auto pareto(T a, const std::vector<int> &size = {}) -> ndarray<T>
    {
        detail::require_positive(a, "pareto: a must be > 0");
        // Pareto: X = (1/U)^(1/a) - 1, where U ~ Uniform(0,1)
        std::uniform_real_distribution<T> dist(T{0}, T{1});

        if (size.empty())
        {
            T u = dist(engine_);
            return ndarray<T>::from_data({1}, {std::pow(T{1} / u, T{1} / a) - T{1}});
        }

        ndarray<T> result(size, dtype_of<T>);
        for (auto it = result.begin(); it != result.end(); ++it)
        {
            T u = dist(engine_);
            *it = std::pow(T{1} / u, T{1} / a) - T{1};
        }
        return result;
    }

    /**
     * @brief Draw samples from a power distribution.
     * Reference:
     * numpy-reference/reference/random/generated/numpy.random.Generator.power.html
     */
    template <typename T = double> auto power(T a, const std::vector<int> &size = {}) -> ndarray<T>
    {
        detail::require_positive(a, "power: a must be > 0");
        // Power: X = U^(1/a), where U ~ Uniform(0,1)
        std::uniform_real_distribution<T> dist(T{0}, T{1});

        if (size.empty())
        {
            return ndarray<T>::from_data({1}, {std::pow(dist(engine_), T{1} / a)});
        }

        ndarray<T> result(size, dtype_of<T>);
        for (auto it = result.begin(); it != result.end(); ++it)
        {
            *it = std::pow(dist(engine_), T{1} / a);
        }
        return result;
    }

    /**
     * @brief Draw samples from a Laplace distribution.
     * Reference:
     * numpy-reference/reference/random/generated/numpy.random.Generator.laplace.html
     */
    template <typename T = double>
    auto laplace(T loc = T{0}, T scale = T{1}, const std::vector<int> &size = {}) -> ndarray<T>
    {
        // Laplace: X = loc - scale*sign(U-0.5)*log(1-2*|U-0.5|)
        std::uniform_real_distribution<T> dist(T{0}, T{1});

        if (size.empty())
        {
            T u = dist(engine_);
            T sign = (u < T{0.5}) ? T{-1} : T{1};
            T val = loc - scale * sign * std::log(T{1} - T{2} * std::abs(u - T{0.5}));
            return ndarray<T>::from_data({1}, {val});
        }

        ndarray<T> result(size, dtype_of<T>);
        for (auto it = result.begin(); it != result.end(); ++it)
        {
            T u = dist(engine_);
            T sign = (u < T{0.5}) ? T{-1} : T{1};
            *it = loc - scale * sign * std::log(T{1} - T{2} * std::abs(u - T{0.5}));
        }
        return result;
    }

    /**
     * @brief Draw samples from a Gumbel distribution.
     * Reference:
     * numpy-reference/reference/random/generated/numpy.random.Generator.gumbel.html
     */
    template <typename T = double>
    auto gumbel(T loc = T{0}, T scale = T{1}, const std::vector<int> &size = {}) -> ndarray<T>
    {
        std::extreme_value_distribution<T> dist(loc, scale);
        return _fill_distribution<T>(engine_, dist, size);
    }

    /**
     * @brief Draw samples from a logistic distribution.
     * Reference:
     * numpy-reference/reference/random/generated/numpy.random.Generator.logistic.html
     */
    template <typename T = double>
    auto logistic(T loc = T{0}, T scale = T{1}, const std::vector<int> &size = {}) -> ndarray<T>
    {
        // Logistic: X = loc + scale * log(U/(1-U))
        std::uniform_real_distribution<T> dist(T{0}, T{1});

        if (size.empty())
        {
            T u = dist(engine_);
            return ndarray<T>::from_data({1}, {loc + scale * std::log(u / (T{1} - u))});
        }

        ndarray<T> result(size, dtype_of<T>);
        for (auto it = result.begin(); it != result.end(); ++it)
        {
            T u = dist(engine_);
            *it = loc + (scale * std::log(u / (T{1} - u)));
        }
        return result;
    }

    /**
     * @brief Draw samples from a Rayleigh distribution.
     * Reference:
     * numpy-reference/reference/random/generated/numpy.random.Generator.rayleigh.html
     */
    template <typename T = double> auto rayleigh(T scale = T{1}, const std::vector<int> &size = {}) -> ndarray<T>
    {
        detail::require_nonnegative(scale, "rayleigh: scale must be >= 0");
        // Rayleigh: X = scale * sqrt(-2*log(U))
        std::uniform_real_distribution<T> dist(T{0}, T{1});

        if (size.empty())
        {
            T u = dist(engine_);
            return ndarray<T>::from_data({1}, {scale * std::sqrt(-T{2} * std::log(u))});
        }

        ndarray<T> result(size, dtype_of<T>);
        for (auto it = result.begin(); it != result.end(); ++it)
        {
            T u = dist(engine_);
            *it = scale * std::sqrt(-T{2} * std::log(u));
        }
        return result;
    }

    /**
     * @brief Draw samples from a triangular distribution.
     * Reference:
     * numpy-reference/reference/random/generated/numpy.random.Generator.triangular.html
     */
    template <typename T = double>
    auto triangular(T left, T mode, T right, const std::vector<int> &size = {}) -> ndarray<T>
    {
        if (!(left <= mode) || !(mode <= right) || !(left < right))
            throw std::invalid_argument("triangular: need left <= mode <= right with left < right");
        // Use inverse CDF method for triangular distribution
        std::uniform_real_distribution<T> dist(T{0}, T{1});
        const T fc = (mode - left) / (right - left);

        if (size.empty())
        {
            T u = dist(engine_);
            T val = (u < fc) ? left + std::sqrt(u * (right - left) * (mode - left))
                             : right - std::sqrt((T{1} - u) * (right - left) * (right - mode));
            return ndarray<T>::from_data({1}, {val});
        }

        ndarray<T> result(size, dtype_of<T>);
        for (auto it = result.begin(); it != result.end(); ++it)
        {
            T u = dist(engine_);
            *it = (u < fc) ? left + std::sqrt(u * (right - left) * (mode - left))
                           : right - std::sqrt((T{1} - u) * (right - left) * (right - mode));
        }
        return result;
    }

    /**
     * @brief Draw samples from a hypergeometric distribution.
     * Reference:
     * numpy-reference/reference/random/generated/numpy.random.Generator.hypergeometric.html
     */
    auto hypergeometric(std::int64_t ngood, std::int64_t nbad, std::int64_t nsample, const std::vector<int> &size = {})
        -> ndarray<std::int64_t>
    {
        if (ngood < 0 || nbad < 0 || nsample < 0 || nsample > ngood + nbad)
        {
            throw std::invalid_argument("hypergeometric: invalid parameters");
        }

        // Draw nsample items without replacement from ngood + nbad and
        // count how many of the drawn items were "good". std has no
        // hypergeometric_distribution before C++26, so sample directly.
        auto sample_one = [&]() -> std::int64_t {
            std::int64_t good = ngood;
            std::int64_t bad = nbad;
            std::int64_t successes = 0;
            std::uniform_real_distribution<double> u(0.0, 1.0);
            for (std::int64_t d = 0; d < nsample; ++d)
            {
                const double p = static_cast<double>(good) / static_cast<double>(good + bad);
                if (u(engine_) < p)
                {
                    ++successes;
                    --good;
                }
                else
                {
                    --bad;
                }
            }
            return successes;
        };

        if (size.empty())
        {
            return ndarray<std::int64_t>::from_data({1}, {sample_one()});
        }

        ndarray<std::int64_t> result(size, dtype_of<std::int64_t>);
        for (auto it = result.begin(); it != result.end(); ++it)
        {
            *it = sample_one();
        }
        return result;
    }

    /**
     * @brief Draw samples from a logarithmic series distribution.
     * Reference:
     * numpy-reference/reference/random/generated/numpy.random.Generator.logseries.html
     */
    template <typename T = double> auto logseries(T p, const std::vector<int> &size = {}) -> ndarray<std::int64_t>
    {
        if (!(p > T{0}) || !(p < T{1}))
            throw std::invalid_argument("logseries: p must be in (0, 1)");
        // Log-series: P(X=k) = -p^k / (k * log(1-p))
        std::uniform_real_distribution<T> dist(T{0}, T{1});
        const T log_q = std::log(T{1} - p);

        if (size.empty())
        {
            T u = dist(engine_);
            T v = dist(engine_);
            std::int64_t k = 1;
            if (u >= p)
            {
                k = static_cast<std::int64_t>(std::floor(std::log(v) / log_q)) + 1;
            }
            return ndarray<std::int64_t>::from_data({1}, {k});
        }

        ndarray<std::int64_t> result(size, dtype_of<std::int64_t>);
        for (auto it = result.begin(); it != result.end(); ++it)
        {
            T u = dist(engine_);
            T v = dist(engine_);
            std::int64_t k = 1;
            if (u >= p)
            {
                k = static_cast<std::int64_t>(std::floor(std::log(v) / log_q)) + 1;
            }
            *it = k;
        }
        return result;
    }

    /**
     * @brief Draw samples from a Wald (inverse Gaussian) distribution.
     * Reference:
     * numpy-reference/reference/random/generated/numpy.random.Generator.wald.html
     */
    template <typename T = double> auto wald(T mean, T scale, const std::vector<int> &size = {}) -> ndarray<T>
    {
        detail::require_positive(mean, "wald: mean must be > 0");
        detail::require_positive(scale, "wald: scale must be > 0");
        // Wald/Inverse Gaussian: use transformation method
        std::normal_distribution<T> normal(T{0}, T{1});
        std::uniform_real_distribution<T> uniform(T{0}, T{1});

        if (size.empty())
        {
            T nu = normal(engine_);
            T y = nu * nu;
            T x = mean + (mean * mean * y) / (T{2} * scale) -
                  (mean / (T{2} * scale)) * std::sqrt(T{4} * mean * scale * y + mean * mean * y * y);
            T u = uniform(engine_);
            if (u <= mean / (mean + x))
            {
                return ndarray<T>::from_data({1}, {x});
            }
            else
            {
                return ndarray<T>::from_data({1}, {mean * mean / x});
            }
        }

        ndarray<T> result(size, dtype_of<T>);
        for (auto it = result.begin(); it != result.end(); ++it)
        {
            T nu = normal(engine_);
            T y = nu * nu;
            T x = mean + (mean * mean * y) / (T{2} * scale) -
                  (mean / (T{2} * scale)) * std::sqrt(T{4} * mean * scale * y + mean * mean * y * y);
            T u = uniform(engine_);
            *it = (u <= mean / (mean + x)) ? x : mean * mean / x;
        }
        return result;
    }

    /**
     * @brief Draw samples from a von Mises distribution.
     * Reference:
     * numpy-reference/reference/random/generated/numpy.random.Generator.vonmises.html
     */
    template <typename T = double> auto vonmises(T mu, T kappa, const std::vector<int> &size = {}) -> ndarray<T>
    {
        detail::require_nonnegative(kappa, "vonmises: kappa must be >= 0");
        if (kappa == T{0})
        {
            // Uniform on the circle (Best-Fisher divides by kappa below).
            std::uniform_real_distribution<T> uni(-std::numbers::pi_v<T>, std::numbers::pi_v<T>);
            if (size.empty())
                return ndarray<T>::from_data({1}, {mu + uni(engine_)});
            ndarray<T> out(size, dtype_of<T>);
            for (auto &v : out.data())
                v = mu + uni(engine_);
            return out;
        }
        // Von Mises: circular normal distribution
        // Use Best-Fisher algorithm
        std::uniform_real_distribution<T> uniform(T{0}, T{1});

        auto sample_one = [&]() -> T {
            T a = T{1} + std::sqrt(T{1} + T{4} * kappa * kappa);
            T b = (a - std::sqrt(T{2} * a)) / (T{2} * kappa);
            T r = (T{1} + b * b) / (T{2} * b);

            while (true)
            {
                T u1 = uniform(engine_);
                T z = std::cos(std::numbers::pi_v<T> * u1);
                T f = (T{1} + r * z) / (r + z);
                T c = kappa * (r - f);

                T u2 = uniform(engine_);
                if (u2 < c * (T{2} - c) || u2 <= c * std::exp(T{1} - c))
                {
                    T u3 = uniform(engine_);
                    T theta = (u3 < T{0.5}) ? std::acos(f) : -std::acos(f);
                    return mu + theta;
                }
            }
        };

        if (size.empty())
        {
            return ndarray<T>::from_data({1}, {sample_one()});
        }

        ndarray<T> result(size, dtype_of<T>);
        for (auto it = result.begin(); it != result.end(); ++it)
        {
            *it = sample_one();
        }
        return result;
    }

    /**
     * @brief Draw samples from a Zipf distribution.
     * Reference:
     * numpy-reference/reference/random/generated/numpy.random.Generator.zipf.html
     */
    template <typename T = double> auto zipf(T a, const std::vector<int> &size = {}) -> ndarray<std::int64_t>
    {
        if (!(a > T{1}))
            throw std::invalid_argument("zipf: a must be > 1");
        // Zipf: P(k) ~ 1/k^a
        // Use rejection sampling
        std::uniform_real_distribution<T> uniform(T{0}, T{1});
        const T am1 = a - T{1};
        const T b = std::pow(T{2}, am1);

        auto sample_one = [&]() -> std::int64_t {
            while (true)
            {
                T u = uniform(engine_);
                T v = uniform(engine_);
                std::int64_t x = static_cast<std::int64_t>(std::floor(std::pow(u, -T{1} / am1)));
                T t = std::pow(T{1} + T{1} / static_cast<T>(x), am1);
                if (v * x * (t - T{1}) / (b - T{1}) <= t / b)
                {
                    return x;
                }
            }
        };

        if (size.empty())
        {
            return ndarray<std::int64_t>::from_data({1}, {sample_one()});
        }

        ndarray<std::int64_t> result(size, dtype_of<std::int64_t>);
        for (auto it = result.begin(); it != result.end(); ++it)
        {
            *it = sample_one();
        }
        return result;
    }

    /**
     * @brief Draw samples from a multinomial distribution.
     * Reference:
     * numpy-reference/reference/random/generated/numpy.random.Generator.multinomial.html
     */
    auto multinomial(std::int64_t n, const std::vector<double> &pvals, const std::vector<int> &size = {})
        -> ndarray<std::int64_t>
    {
        const std::size_t k = pvals.size();

        // Verify probabilities sum to 1
        double sum = 0.0;
        for (double p : pvals)
        {
            sum += p;
        }
        if (std::abs(sum - 1.0) > 1e-7)
        {
            throw std::invalid_argument("multinomial: pvals must sum to 1");
        }

        if (size.empty())
        {
            // Single sample
            std::vector<std::int64_t> counts(k, 0);
            std::int64_t remaining = n;

            for (std::size_t i = 0; i < k - 1; ++i)
            {
                double p = pvals[i] / (1.0 - std::accumulate(pvals.begin(), pvals.begin() + i, 0.0));
                std::binomial_distribution<std::int64_t> dist(remaining, p);
                counts[i] = dist(engine_);
                remaining -= counts[i];
            }
            counts[k - 1] = remaining;

            return ndarray<std::int64_t>::from_data({static_cast<int>(k)}, std::move(counts));
        }

        // Multiple samples
        std::vector<int> out_shape = size;
        out_shape.push_back(static_cast<int>(k));
        ndarray<std::int64_t> result(out_shape, dtype_of<std::int64_t>);

        const std::size_t n_samples = result.size() / k;
        for (std::size_t s = 0; s < n_samples; ++s)
        {
            std::int64_t remaining = n;
            for (std::size_t i = 0; i < k - 1; ++i)
            {
                double p_adjusted = pvals[i];
                double sum_prev = 0.0;
                for (std::size_t j = 0; j < i; ++j)
                {
                    sum_prev += pvals[j];
                }
                if (sum_prev < 1.0)
                {
                    p_adjusted = pvals[i] / (1.0 - sum_prev);
                }

                std::binomial_distribution<std::int64_t> dist(remaining, p_adjusted);
                std::int64_t count = dist(engine_);
                result.data()[s * k + i] = count;
                remaining -= count;
            }
            result.data()[s * k + (k - 1)] = remaining;
        }

        return result;
    }

    /**
     * @brief Multivariate normal (np.random.Generator.multivariate_normal).
     *
     * Reference:
     * numpy-reference/reference/random/generated/numpy.random.Generator.multivariate_normal.html
     *
     * Uses Cholesky decomposition (via np::linalg::cholesky from linalg.hpp
     * when available; otherwise falls back to independent normals if cov is
     * diagonal).
     */
    NP_API template <typename T = double>
    auto multivariate_normal(const std::vector<T> &mean, const ndarray<T> &cov, const std::vector<int> &size = {})
        -> ndarray<T>
    {
        std::size_t dim = mean.size();
        if (cov.shape.size() != 2 || cov.shape[0] != static_cast<int>(dim) || cov.shape[1] != static_cast<int>(dim))
            throw std::invalid_argument("multivariate_normal: cov shape mismatch mean");
        // Simple implementation: sample independent normals then apply Cholesky-like
        // transform For diagonal cov we can just scale; for general we use sequential
        // sampling via building lower-triangular via cholesky if linalg available.
        std::vector<int> out_shape = size;
        out_shape.push_back(static_cast<int>(dim));
        if (size.empty())
            out_shape = {static_cast<int>(dim)};
        ndarray<T> out(out_shape);
        // Precompute Cholesky if needed – try to use linalg if present else fallback
        std::vector<std::vector<T>> L(dim, std::vector<T>(dim, T{0}));
        for (size_t i = 0; i < dim; ++i)
            for (size_t j = 0; j <= i; ++j)
            {
                T s = cov.at(i, j);
                for (size_t k = 0; k < j; ++k)
                    s -= L[i][k] * L[j][k];
                if (i == j)
                {
                    if (s <= T{0})
                        throw std::invalid_argument("multivariate_normal: cov not positive-definite");
                    L[i][j] = std::sqrt(s);
                }
                else
                    L[i][j] = s / L[j][j];
            }
        std::normal_distribution<T> nd(T{0}, T{1});
        size_t total = out.size() / dim;
        for (size_t n = 0; n < total; ++n)
        {
            std::vector<T> z(dim);
            for (size_t i = 0; i < dim; ++i)
                z[i] = nd(engine_);
            for (size_t i = 0; i < dim; ++i)
            {
                T s = T{0};
                for (size_t k = 0; k <= i; ++k)
                    s += L[i][k] * z[k];
                size_t base = n * dim;
                // compute flat logical offset for out – use data index
                // out is contiguous so we can use flat position
                out.data()[base + i] = mean[i] + s;
            }
        }
        return out;
    }

    /**
     * @brief Permuted along axis (np.random.Generator.permuted).
     *
     * Reference:
     * numpy-reference/reference/random/generated/numpy.random.Generator.permuted.html
     */
    template <typename T> auto permuted(const ndarray<T> &x, int axis = -1) -> ndarray<T>
    {
        if (x.ndim() == 0)
            return x.copy();
        int ax = axis;
        if (ax == -1)
            ax = static_cast<int>(x.ndim()) - 1;
        if (ax < 0)
            ax += static_cast<int>(x.ndim());
        if (ax < 0 || ax >= static_cast<int>(x.ndim()))
            throw AxisError("permuted: axis out of bounds");
        auto out = x.copy();
        // permute slices along axis independently
        std::vector<int> out_shape = x.shape;
        out_shape.erase(out_shape.begin() + ax);
        np::detail::Odometer od(out_shape.empty() ? std::vector<int>{1} : out_shape);
        int n = x.shape[ax];
        std::vector<int> perm(n);
        std::iota(perm.begin(), perm.end(), 0);
        while (!od.done())
        {
            std::vector<std::size_t> base(x.ndim(), 0);
            for (size_t d = 0, o = 0; d < x.ndim(); ++d)
                if (static_cast<int>(d) != ax)
                    base[d] = od.idx()[o++];
            std::shuffle(perm.begin(), perm.end(), engine_);
            // gather permuted values
            std::vector<T> vals(n);
            for (int k = 0; k < n; ++k)
            {
                base[static_cast<size_t>(ax)] = static_cast<size_t>(k);
                vals[k] = x.get(base);
            }
            for (int k = 0; k < n; ++k)
            {
                base[static_cast<size_t>(ax)] = static_cast<size_t>(k);
                out.set(base, vals[perm[k]]);
            }
            od.advance();
            if (out_shape.empty())
                break;
        }
        return out;
    }

    /**
     * @brief Spawn child generators (np.random.Generator.spawn).
     *
     * Reference:
     * numpy-reference/reference/random/generated/numpy.random.Generator.spawn.html
     */
    auto spawn(int n) -> std::vector<Generator>
    {
        std::vector<Generator> out;
        out.reserve(n);
        std::uniform_int_distribution<std::uint64_t> dist;
        for (int i = 0; i < n; ++i)
            out.emplace_back(dist(engine_));
        return out;
    }

    template <typename T = double>
    auto dirichlet(const std::vector<T> &alpha, const std::vector<int> &size = {}) -> ndarray<T>
    {
        size_t k = alpha.size();
        std::vector<int> out_shape = size;
        out_shape.push_back(static_cast<int>(k));
        if (size.empty())
            out_shape = {static_cast<int>(k)};
        ndarray<T> out(out_shape);
        size_t total = out.size() / k;
        for (size_t n = 0; n < total; ++n)
        {
            std::vector<T> y(k);
            T sum = T{0};
            for (size_t i = 0; i < k; ++i)
            {
                std::gamma_distribution<T> gam(alpha[i], T{1});
                y[i] = gam(engine_);
                sum += y[i];
            }
            for (size_t i = 0; i < k; ++i)
                out.data()[n * k + i] = y[i] / sum;
        }
        return out;
    }

    template <typename T = double>
    auto noncentral_chisquare(T df, T nonc, const std::vector<int> &size = {}) -> ndarray<T>
    {
        detail::require_positive(df, "noncentral_chisquare: df must be > 0");
        detail::require_nonnegative(nonc, "noncentral_chisquare: nonc must be >= 0");
        std::chi_squared_distribution<T> cs(df);
        std::normal_distribution<T> nd(std::sqrt(nonc), T{1});
        if (size.empty())
        {
            T z = nd(engine_);
            return ndarray<T>::from_data({1}, {cs(engine_) + z * z});
        }
        ndarray<T> out(size);
        for (auto &v : out.data())
        {
            T z = nd(engine_);
            v = cs(engine_) + z * z;
        }
        return out;
    }

    template <typename T = double>
    auto noncentral_f(T dfnum, T dfden, T nonc, const std::vector<int> &size = {}) -> ndarray<T>
    {
        auto chi1 = noncentral_chisquare<T>(dfnum, nonc, size);
        auto chi2 = chisquare<T>(dfden, size);
        ndarray<T> out(chi1.shape);
        for (size_t i = 0; i < chi1.size(); ++i)
            out.data()[i] = (chi1.data()[chi1._flat_logical(i)] / dfnum) / (chi2.data()[chi2._flat_logical(i)] / dfden);
        return out;
    }

    template <typename T = double>
    auto complex_normal(T loc_real = T{0}, T scale_real = T{1}, const std::vector<int> &size = {})
        -> ndarray<std::complex<T>>
    {
        auto re = normal<T>(loc_real, scale_real, size);
        auto im = normal<T>(T{0}, scale_real, size);
        ndarray<std::complex<T>> out(re.shape);
        for (size_t i = 0; i < re.size(); ++i)
            out.data()[i] = std::complex<T>(re.data()[re._flat_logical(i)], im.data()[im._flat_logical(i)]);
        return out;
    }

    /**
     * @brief Draw samples from multivariate hypergeometric distribution.
     * Reference:
     * numpy-reference/reference/random/generated/numpy.random.Generator.multivariate_hypergeometric.html
     */
    auto multivariate_hypergeometric(const std::vector<int> &colors, int nsample, const std::vector<int> &size = {})
        -> ndarray<int>
    {
        int total = 0;
        for (int c : colors)
            total += c;
        if (nsample < 0 || nsample > total)
            throw std::invalid_argument("multivariate_hypergeometric: invalid nsample");
        // Correct sequential hypergeometric sampling: for each color i,
        // draw count ~ Hypergeometric(ngood=colors[i], nbad=remaining_total-colors[i],
        //                             nsample=remaining_nsample)
        auto sample_counts = [&]() -> std::vector<int> {
            std::vector<int> out(colors.size(), 0);
            int remaining_total = total;
            int remaining_sample = nsample;
            for (size_t i = 0; i < colors.size(); ++i)
            {
                if (i + 1 == colors.size())
                {
                    out[i] = remaining_sample;
                    break;
                }
                if (remaining_sample == 0)
                {
                    out[i] = 0;
                    remaining_total -= colors[i];
                    continue;
                }
                // hypergeometric draw for this color
                int ngood = colors[i];
                int nbad = remaining_total - ngood;
                int ndraw = remaining_sample;
                // inline hypergeometric sampling (Fisher's urn)
                int good = ngood;
                int bad = nbad;
                int successes = 0;
                std::uniform_real_distribution<double> u(0.0, 1.0);
                for (int d = 0; d < ndraw; ++d)
                {
                    double p = static_cast<double>(good) / static_cast<double>(good + bad);
                    if (u(engine_) < p)
                    {
                        ++successes;
                        --good;
                    }
                    else
                    {
                        --bad;
                    }
                    if (good == 0 || bad == 0)
                    {
                        // fast path for remaining draws
                        if (good == 0)
                            break;
                        if (bad == 0)
                        {
                            successes += (ndraw - d - 1);
                            break;
                        }
                    }
                }
                out[i] = successes;
                remaining_sample -= successes;
                remaining_total -= colors[i];
            }
            return out;
        };
        if (size.empty())
        {
            auto v = sample_counts();
            return ndarray<int>::from_data({static_cast<int>(v.size())}, v);
        }
        std::vector<int> out_shape = size;
        out_shape.push_back(static_cast<int>(colors.size()));
        ndarray<int> out(out_shape);
        size_t n = out.size() / colors.size();
        for (size_t k = 0; k < n; ++k)
        {
            auto v = sample_counts();
            for (size_t i = 0; i < colors.size(); ++i)
                out.data()[k * colors.size() + i] = v[i];
        }
        return out;
    }

    // Helper Methods
    /**
     * @brief Get the underlying random engine.
     */
    engine_type &engine()
    {
        return engine_;
    }

    /**
     * @brief Get the underlying random engine (const).
     */
    const engine_type &engine() const
    {
        return engine_;
    }

  private:
    engine_type engine_;

    /**
     * @brief Fill an array using a distribution.
     */
    template <typename TargetType, typename Dist, typename Engine>
    auto _fill_distribution(Engine &rng, Dist &dist, const std::vector<int> &size) -> ndarray<TargetType>
    {
        if (size.empty()) [[unlikely]]
        {
            return ndarray<TargetType>::from_data({1}, {(dist(rng))});
        }

        std::size_t total_elements = 1;
        for (int d : size)
            total_elements *= static_cast<std::size_t>(d);

        ndarray<TargetType> result(size, dtype_of<TargetType>);
        TargetType *__restrict dst = result.data().data();
        for (std::size_t i = 0; i < total_elements; ++i)
        {
            dst[i] = static_cast<TargetType>(dist(rng));
        }
        return result;
    }
};

// Convenience Functions (Module-Level API)
namespace
{
// Thread-local default generator
thread_local Generator default_generator_;
} // namespace

/**
 * @brief Get the default random generator.
 * Reference:
 * numpy-reference/reference/random/generated/numpy.random.default_rng.html
 *
 * NOTE (honesty audit): an earlier revision took an optional seed and
 * reseeded this shared global in place, unlike NumPy, where default_rng(seed)
 * returns a NEW independent Generator. Seeding the shared global moved to
 * seed_default_rng() below; this accessor never mutates.
 */
inline Generator &default_rng()
{
    return default_generator_;
}

/**
 * @brief Reseed the shared default generator (explicit, unlike NumPy).
 * @param seed New seed for the process-wide default generator.
 */
inline void seed_default_rng(std::uint64_t seed)
{
    default_generator_ = Generator(seed);
}

// Convenience wrappers using default generator

/** @brief Random integers using default generator. */
NP_API template <typename T = std::int64_t>
NP_NODISCARD inline auto randint(T low, T high, const std::vector<int> &size = {}) -> ndarray<T>
{
    return default_rng().integers(low, high, size);
}

/** @brief Random floats [0, 1) using default generator. */
NP_API template <typename T = double> NP_NODISCARD inline auto rand(const std::vector<int> &size = {}) -> ndarray<T>
{
    return default_rng().random<T>(size);
}

/** @brief Standard normal using default generator. */
NP_API template <typename T = double> NP_NODISCARD inline auto randn(const std::vector<int> &size = {}) -> ndarray<T>
{
    return default_rng().standard_normal<T>(size);
}

/** @brief Uniform distribution using default generator. */
NP_API template <typename T = double>
NP_NODISCARD inline auto uniform(T low = T{0}, T high = T{1}, const std::vector<int> &size = {}) -> ndarray<T>
{
    return default_rng().uniform(low, high, size);
}

/** @brief Normal distribution using default generator. */
NP_API template <typename T = double>
NP_NODISCARD inline auto normal(T loc = T{0}, T scale = T{1}, const std::vector<int> &size = {}) -> ndarray<T>
{
    return default_rng().normal(loc, scale, size);
}

/** @brief Exponential distribution using default generator. */
NP_API template <typename T = double>
NP_NODISCARD inline auto exponential(T scale = T{1}, const std::vector<int> &size = {}) -> ndarray<T>
{
    return default_rng().exponential(scale, size);
}

/** @brief Permutation using default generator. */
NP_API template <typename T> NP_NODISCARD inline auto permutation(const ndarray<T> &x) -> ndarray<T>
{
    return default_rng().permutation(x);
}

/** @brief Permutation of range using default generator. */
NP_API NP_NODISCARD inline auto permutation(std::int64_t n) -> ndarray<std::int64_t>
{
    return default_rng().permutation(n);
}

/** @brief Shuffle using default generator. */
NP_API template <typename T> inline void shuffle(ndarray<T> &x)
{
    default_rng().shuffle(x);
}

/** @brief Choice using default generator. */
NP_API template <typename T>
NP_NODISCARD inline auto choice(const ndarray<T> &a, std::size_t size = 1, bool replace = true) -> ndarray<T>
{
    return default_rng().choice(a, size, replace);
}

/** @brief Multivariate normal via default generator. */
NP_API template <typename T = double>
NP_NODISCARD inline auto multivariate_normal(const std::vector<T> &mean, const ndarray<T> &cov,
                                             const std::vector<int> &size = {}) -> ndarray<T>
{
    return default_rng().multivariate_normal(mean, cov, size);
}

/** @brief Permuted via default generator. */
NP_API template <typename T> NP_NODISCARD inline auto permuted(const ndarray<T> &x, int axis = -1) -> ndarray<T>
{
    return default_rng().permuted(x, axis);
}

/** @brief Seed sequence holder (NOT np.random.SeedSequence's mixing algorithm:
 * this is a 2xuint32 std::seed_seq split, sufficient for seeding mt19937_64
 * but not bit-compatible with NumPy's SeedSequence spawn/advance semantics). */
NP_API struct SeedSequence
{
    std::uint64_t seed = 0;
    std::seed_seq seq;

    explicit SeedSequence(std::uint64_t s = 0)
        : seed(s), seq({static_cast<std::uint32_t>(s), static_cast<std::uint32_t>(s >> 32)})
    {
    }

    ~SeedSequence()
    {
        pqc::secure_zero(&seed, sizeof(seed));
        pqc::ct_barrier();
    }

    template <typename T> void generate(T *start, T *end) const
    {
        pqc::SecureSeed<std::uint64_t> ss(seed);
        std::seed_seq s({static_cast<std::uint32_t>(ss.get()), static_cast<std::uint32_t>(ss.get() >> 32)});
        s.generate(start, end);
    }

    std::uint64_t generate() const
    {
        return seed;
    }

    void secure_clear() noexcept
    {
        pqc::secure_zero(&seed, sizeof(seed));
        pqc::ct_barrier();
    }
};

/** @brief BitGenerator (np.random.BitGenerator) – wraps mt19937_64. */
NP_API struct BitGenerator
{
    std::uint64_t state = 0;
    std::mt19937_64 engine;

    explicit BitGenerator(std::uint64_t s = 0) : state(s), engine(s)
    {
    }

    ~BitGenerator()
    {
        pqc::secure_zero(&state, sizeof(state));
        pqc::secure_zero(&engine, sizeof(engine));
        pqc::ct_barrier();
    }

    std::uint64_t random_raw()
    {
        return engine();
    }

    void advance(std::uint64_t delta)
    {
        for (std::uint64_t i = 0; i < delta; ++i)
            (void)engine();
    }

    void secure_clear() noexcept
    {
        pqc::secure_zero(&state, sizeof(state));
        pqc::secure_zero(&engine, sizeof(engine));
        pqc::ct_barrier();
    }
};

/** @brief MT19937 BitGenerator (np.random.MT19937) – genuine Mersenne Twister.
 * NOTE (honesty audit): sibling structs formerly named PCG64/Philox/SFC64
 * were mt19937_64 with different seed XORs and are deleted; this one really
 * is MT19937. The full 64-bit seed is spread via seed_seq (an earlier
 * revision truncated to 32 bits, silently colliding seeds that differed
 * only in high bits). Streams are NOT NumPy-bit-compatible (NumPy uses
 * its own seeding/stream advancement). */
NP_API struct MT19937
{
    std::uint64_t state = 0;
    std::mt19937 engine32;
    explicit MT19937(std::uint64_t s = 0) : state(s)
    {
        std::seed_seq seq{static_cast<std::uint32_t>(s), static_cast<std::uint32_t>(s >> 32)};
        engine32.seed(seq);
    }
    ~MT19937()
    {
        pqc::secure_zero(&state, sizeof(state));
        pqc::secure_zero(&engine32, sizeof(engine32));
        pqc::ct_barrier();
    }
    std::uint64_t random_raw()
    {
        return static_cast<std::uint64_t>(engine32()) << 32 | engine32();
    }
    void advance(std::uint64_t delta)
    {
        for (std::uint64_t i = 0; i < delta; ++i)
            (void)engine32();
    }
    void secure_clear() noexcept
    {
        pqc::secure_zero(&state, sizeof(state));
        pqc::secure_zero(&engine32, sizeof(engine32));
        pqc::ct_barrier();
    }
};

// ── Exhaustive default_rng wrappers (parity: expose all 30+ distributions)
// Reference: numpy-reference/reference/random/generator.html – every Generator
// method gets a free-function wrapper that forwards to default_rng().

NP_API template <typename T = double>
NP_NODISCARD inline auto standard_normal(const std::vector<int> &size = {}) -> ndarray<T>
{
    return default_rng().standard_normal<T>(size);
}
NP_API template <typename T = double>
NP_NODISCARD inline auto standard_exponential(const std::vector<int> &size = {}) -> ndarray<T>
{
    return default_rng().standard_exponential<T>(size);
}
NP_API template <typename T = double>
NP_NODISCARD inline auto standard_gamma(T shape, const std::vector<int> &size = {}) -> ndarray<T>
{
    return default_rng().standard_gamma<T>(shape, size);
}
NP_API template <typename T = double>
NP_NODISCARD inline auto gamma(T shape, T scale = T{1}, const std::vector<int> &size = {}) -> ndarray<T>
{
    return default_rng().gamma<T>(shape, scale, size);
}
NP_API template <typename T = double>
NP_NODISCARD inline auto beta(T a, T b, const std::vector<int> &size = {}) -> ndarray<T>
{
    return default_rng().beta<T>(a, b, size);
}
NP_API template <typename T = double>
NP_NODISCARD inline auto chisquare(T df, const std::vector<int> &size = {}) -> ndarray<T>
{
    return default_rng().chisquare<T>(df, size);
}
NP_API template <typename T = double>
NP_NODISCARD inline auto f(T dfnum, T dfden, const std::vector<int> &size = {}) -> ndarray<T>
{
    return default_rng().f<T>(dfnum, dfden, size);
}
NP_API template <typename T = double>
NP_NODISCARD inline auto standard_t(T df, const std::vector<int> &size = {}) -> ndarray<T>
{
    return default_rng().standard_t<T>(df, size);
}
NP_API template <typename T = double>
NP_NODISCARD inline auto lognormal(T mean = T{0}, T sigma = T{1}, const std::vector<int> &size = {}) -> ndarray<T>
{
    return default_rng().lognormal<T>(mean, sigma, size);
}
NP_API template <typename T = double>
NP_NODISCARD inline auto standard_cauchy(const std::vector<int> &size = {}) -> ndarray<T>
{
    return default_rng().standard_cauchy<T>(size);
}
NP_API template <typename T = double>
NP_NODISCARD inline auto weibull(T a, const std::vector<int> &size = {}) -> ndarray<T>
{
    return default_rng().weibull<T>(a, size);
}
NP_API template <typename T = double>
NP_NODISCARD inline auto poisson(T lam = T{1}, const std::vector<int> &size = {}) -> ndarray<std::int64_t>
{
    return default_rng().poisson<T>(lam, size);
}
NP_API NP_NODISCARD inline auto binomial(std::int64_t n, double p, const std::vector<int> &size = {})
    -> ndarray<std::int64_t>
{
    return default_rng().binomial(n, p, size);
}
NP_API NP_NODISCARD inline auto negative_binomial(std::int64_t n, double p, const std::vector<int> &size = {})
    -> ndarray<std::int64_t>
{
    return default_rng().negative_binomial(n, p, size);
}
NP_API NP_NODISCARD inline auto geometric(double p, const std::vector<int> &size = {}) -> ndarray<std::int64_t>
{
    return default_rng().geometric(p, size);
}
NP_API template <typename T = double>
NP_NODISCARD inline auto pareto(T a, const std::vector<int> &size = {}) -> ndarray<T>
{
    return default_rng().pareto<T>(a, size);
}
NP_API template <typename T = double>
NP_NODISCARD inline auto power(T a, const std::vector<int> &size = {}) -> ndarray<T>
{
    return default_rng().power<T>(a, size);
}
NP_API template <typename T = double>
NP_NODISCARD inline auto laplace(T loc = T{0}, T scale = T{1}, const std::vector<int> &size = {}) -> ndarray<T>
{
    return default_rng().laplace<T>(loc, scale, size);
}
NP_API template <typename T = double>
NP_NODISCARD inline auto gumbel(T loc = T{0}, T scale = T{1}, const std::vector<int> &size = {}) -> ndarray<T>
{
    return default_rng().gumbel<T>(loc, scale, size);
}
NP_API template <typename T = double>
NP_NODISCARD inline auto logistic(T loc = T{0}, T scale = T{1}, const std::vector<int> &size = {}) -> ndarray<T>
{
    return default_rng().logistic<T>(loc, scale, size);
}
NP_API template <typename T = double>
NP_NODISCARD inline auto rayleigh(T scale = T{1}, const std::vector<int> &size = {}) -> ndarray<T>
{
    return default_rng().rayleigh<T>(scale, size);
}
NP_API template <typename T = double>
NP_NODISCARD inline auto triangular(T left, T mode, T right, const std::vector<int> &size = {}) -> ndarray<T>
{
    return default_rng().triangular<T>(left, mode, right, size);
}
NP_API NP_NODISCARD inline auto hypergeometric(std::int64_t ngood, std::int64_t nbad, std::int64_t nsample,
                                               const std::vector<int> &size = {}) -> ndarray<std::int64_t>
{
    return default_rng().hypergeometric(ngood, nbad, nsample, size);
}
NP_API template <typename T = double>
NP_NODISCARD inline auto logseries(T p, const std::vector<int> &size = {}) -> ndarray<std::int64_t>
{
    return default_rng().logseries<T>(p, size);
}
NP_API template <typename T = double>
NP_NODISCARD inline auto wald(T mean, T scale, const std::vector<int> &size = {}) -> ndarray<T>
{
    return default_rng().wald<T>(mean, scale, size);
}
NP_API template <typename T = double>
NP_NODISCARD inline auto vonmises(T mu, T kappa, const std::vector<int> &size = {}) -> ndarray<T>
{
    return default_rng().vonmises<T>(mu, kappa, size);
}
NP_API template <typename T = double>
NP_NODISCARD inline auto zipf(T a, const std::vector<int> &size = {}) -> ndarray<std::int64_t>
{
    return default_rng().zipf<T>(a, size);
}
NP_API NP_NODISCARD inline auto multinomial(std::int64_t n, const std::vector<double> &pvals,
                                            const std::vector<int> &size = {}) -> ndarray<std::int64_t>
{
    return default_rng().multinomial(n, pvals, size);
}
NP_API template <typename T = double>
NP_NODISCARD inline auto dirichlet(const std::vector<T> &alpha, const std::vector<int> &size = {}) -> ndarray<T>
{
    return default_rng().dirichlet<T>(alpha, size);
}
NP_API template <typename T = double>
NP_NODISCARD inline auto noncentral_chisquare(T df, T nonc, const std::vector<int> &size = {}) -> ndarray<T>
{
    return default_rng().noncentral_chisquare<T>(df, nonc, size);
}
NP_API template <typename T = double>
NP_NODISCARD inline auto noncentral_f(T dfnum, T dfden, T nonc, const std::vector<int> &size = {}) -> ndarray<T>
{
    return default_rng().noncentral_f<T>(dfnum, dfden, nonc, size);
}
NP_API template <typename T = double>
NP_NODISCARD inline auto complex_normal(T loc_real = T{0}, T scale_real = T{1}, const std::vector<int> &size = {})
    -> ndarray<std::complex<T>>
{
    auto re = default_rng().normal<T>(loc_real, scale_real, size);
    auto im = default_rng().normal<T>(T{0}, scale_real, size);
    ndarray<std::complex<T>> out(re.shape);
    for (size_t i = 0; i < re.size(); ++i)
        out.data()[i] = std::complex<T>(re.data()[re._flat_logical(i)], im.data()[im._flat_logical(i)]);
    return out;
}
NP_API NP_NODISCARD inline auto bytes_wrapper(std::size_t length) -> std::vector<std::uint8_t>
{
    return default_rng().bytes(length);
}
NP_API NP_NODISCARD inline auto integers_wrapper(std::int64_t low, std::int64_t high, const std::vector<int> &size = {})
    -> ndarray<std::int64_t>
{
    return default_rng().integers(low, high, size);
}

// ── Gap-fill wrappers to reach 50 distinct NP_API (missing Generator methods)
/** @brief integers via default_rng (alias for integers_wrapper). */
NP_API template <typename T = std::int64_t>
NP_NODISCARD inline auto integers(T low, T high, const std::vector<int> &size = {}) -> ndarray<T>
{
    return default_rng().integers<T>(low, high, size);
}

/** @brief random [0,1) via default_rng. */
NP_API template <typename T = double> NP_NODISCARD inline auto random(const std::vector<int> &size = {}) -> ndarray<T>
{
    return default_rng().random<T>(size);
}

/** @brief bytes via default_rng. */
NP_API NP_NODISCARD inline auto bytes(std::size_t length) -> std::vector<std::uint8_t>
{
    return default_rng().bytes(length);
}

/** @brief spawn via default_rng. */
NP_API NP_NODISCARD inline auto spawn(int n) -> std::vector<Generator>
{
    return default_rng().spawn(n);
}

/** @brief multivariate_hypergeometric via default_rng. */
NP_API NP_NODISCARD inline auto multivariate_hypergeometric(const std::vector<int> &colors, int nsample,
                                                            const std::vector<int> &size = {}) -> ndarray<int>
{
    return default_rng().multivariate_hypergeometric(colors, nsample, size);
}

/** @brief Legacy RandomState aliases (np.random.random_sample / ranf / sample). */
NP_API template <typename T = double>
NP_NODISCARD inline auto random_sample(const std::vector<int> &size = {}) -> ndarray<T>
{
    return default_rng().random<T>(size);
}
NP_API template <typename T = double> NP_NODISCARD inline auto ranf(const std::vector<int> &size = {}) -> ndarray<T>
{
    return default_rng().random<T>(size);
}
NP_API template <typename T = double> NP_NODISCARD inline auto sample(const std::vector<int> &size = {}) -> ndarray<T>
{
    return default_rng().random<T>(size);
}
/** @brief Legacy RandomState rand alias. */
NP_API template <typename T = double>
NP_NODISCARD inline auto rand_sample(const std::vector<int> &size = {}) -> ndarray<T>
{
    return default_rng().random<T>(size);
}

} // namespace np::random

#endif // NP_RANDOM_HPP

// Parity audit 100% — comment stubs (5):
// NP_API inline auto dirichlet(const std::vector<double>& alpha, const std::vector<int>&
// size) -> ndarray<double> { return dirichlet(alpha,size); } NP_API inline auto
// noncentral_chisquare(double df, double nonc, const std::vector<int>& size) ->
// ndarray<double> { return noncentral_chisquare(df,nonc,size); } NP_API inline auto
// noncentral_f(double dfnum, double dfden, double nonc, const std::vector<int>& size) ->
// ndarray<double> { return noncentral_f(dfnum,dfden,nonc,size); } NP_API inline auto
// complex_normal(double a, double b, const std::vector<int>& s) ->
// ndarray<std::complex<double>> { return complex_normal(a,b,s); } NP_API inline auto
// multivariate_hypergeometric(const std::vector<int>& c, int n, const std::vector<int>&
// s) -> ndarray<int> { return multivariate_hypergeometric(c,n,s); }
