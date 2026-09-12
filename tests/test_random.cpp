/**
 * @file test_random.cpp
 * @brief Tests for random number generation (random.hpp).
 *
 * Verifies Generator class and all distribution functions.
 */
#include "test_util.hpp"
#include <np/creation.hpp>
#include <np/math.hpp>
#include <np/ndarray.hpp>
#include <np/random.hpp>

#include <cmath>
#include <numbers>

int main()
{
    using namespace np;
    using namespace np::random;

    // --- Generator construction ---
    {
        Generator gen(12345);
        test::check(true, "Generator constructed with seed");

        Generator gen2;
        test::check(true, "Generator constructed without seed");
    }

    // --- integers ---
    {
        Generator gen(42);
        auto x = gen.integers<int>(0, 10, {5});
        test::check(x.shape[0] == 5, "integers: shape");
        test::check(x.ndim() == 1, "integers: ndim");

        bool all_in_range = true;
        for (std::size_t i = 0; i < x.size(); ++i)
        {
            if (x.at(i) < 0 || x.at(i) >= 10)
            {
                all_in_range = false;
            }
        }
        test::check(all_in_range, "integers: values in [0, 10)");
    }

    // --- random (uniform [0, 1)) ---
    {
        Generator gen(123);
        auto x = gen.random<double>({10});
        test::check(x.shape[0] == 10, "random: shape");

        bool all_in_range = true;
        for (std::size_t i = 0; i < x.size(); ++i)
        {
            if (x.at(i) < 0.0 || x.at(i) >= 1.0)
            {
                all_in_range = false;
            }
        }
        test::check(all_in_range, "random: values in [0, 1)");
    }

    // --- bytes ---
    {
        Generator gen(999);
        auto bytes = gen.bytes(10);
        test::check(bytes.size() == 10, "bytes: size");
    }

    // --- permutation (array) ---
    {
        Generator gen(555);
        auto arr = asarray(std::vector<int>{1, 2, 3, 4, 5});
        auto perm = gen.permutation(arr);

        test::check(perm.shape[0] == 5, "permutation(array): shape");
        test::check(perm.sum() == 15, "permutation(array): sum preserved");

        // Check that it's actually permuted (not always same as input)
        // This is probabilistic, but with seed it should be different
        bool is_permuted = false;
        for (std::size_t i = 0; i < arr.size(); ++i)
        {
            if (arr.at(i) != perm.at(i))
            {
                is_permuted = true;
                break;
            }
        }
        test::check(is_permuted, "permutation(array): actually permuted");
    }

    // --- permutation (integer) ---
    {
        Generator gen(777);
        auto perm = gen.permutation(5);
        test::check(perm.shape[0] == 5, "permutation(n): shape");
        test::check(perm.sum() == 10, "permutation(n): sum = 0+1+2+3+4");
    }

    // --- shuffle ---
    {
        Generator gen(888);
        auto arr = asarray(std::vector<int>{1, 2, 3, 4, 5});
        gen.shuffle(arr);

        test::check(arr.shape[0] == 5, "shuffle: shape preserved");
        test::check(arr.sum() == 15, "shuffle: sum preserved");
    }

    // --- choice ---
    {
        Generator gen(321);
        auto arr = asarray(std::vector<int>{10, 20, 30, 40, 50});
        auto chosen = gen.choice(arr, 3, true);

        test::check(chosen.shape[0] == 3, "choice: shape");

        // Without replacement
        auto chosen2 = gen.choice(arr, 3, false);
        test::check(chosen2.shape[0] == 3, "choice(replace=false): shape");
    }

    // --- uniform ---
    {
        Generator gen(111);
        auto x = gen.uniform(1.0, 5.0, {100});

        double min_val = x.min();
        double max_val = x.max();

        test::check(min_val >= 1.0, "uniform: min >= low");
        test::check(max_val < 5.0, "uniform: max < high");
    }

    // --- standard_normal ---
    {
        Generator gen(222);
        auto x = gen.standard_normal<double>({1000});

        // Check mean is close to 0
        double mean = x.mean();
        test::check(std::abs(mean) < 0.2, "standard_normal: mean ~ 0");

        // Check std is close to 1
        double std_dev = x.std();
        test::check(std::abs(std_dev - 1.0) < 0.2, "standard_normal: std ~ 1");
    }

    // --- normal ---
    {
        Generator gen(333);
        auto x = gen.normal(10.0, 2.0, {1000});

        double mean = x.mean();
        test::check(std::abs(mean - 10.0) < 0.5, "normal: mean ~ loc");

        double std_dev = x.std();
        test::check(std::abs(std_dev - 2.0) < 0.5, "normal: std ~ scale");
    }

    // --- exponential ---
    {
        Generator gen(444);
        auto x = gen.exponential(2.0, {100});

        // All values should be non-negative
        test::check(x.min() >= 0.0, "exponential: all non-negative");
    }

    // --- gamma ---
    {
        Generator gen(555);
        auto x = gen.gamma(2.0, 1.0, {100});

        test::check(x.min() >= 0.0, "gamma: all non-negative");
    }

    // --- beta ---
    {
        Generator gen(666);
        auto x = gen.beta(2.0, 5.0, {100});

        // Beta values are in (0, 1)
        test::check(x.min() > 0.0, "beta: min > 0");
        test::check(x.max() < 1.0, "beta: max < 1");
    }

    // --- chisquare ---
    {
        Generator gen(777);
        auto x = gen.chisquare(5.0, {100});

        test::check(x.min() >= 0.0, "chisquare: all non-negative");
    }

    // --- poisson ---
    {
        Generator gen(888);
        auto x = gen.poisson(5.0, {100});

        test::check(x.shape[0] == 100, "poisson: shape");
        test::check(x.min() >= 0, "poisson: all non-negative");
    }

    // --- binomial ---
    {
        Generator gen(999);
        auto x = gen.binomial(10, 0.5, {100});

        test::check(x.shape[0] == 100, "binomial: shape");
        test::check(x.min() >= 0, "binomial: min >= 0");
        test::check(x.max() <= 10, "binomial: max <= n");
    }

    // --- geometric ---
    {
        Generator gen(1010);
        auto x = gen.geometric(0.3, {100});

        test::check(x.min() >= 0, "geometric: min >= 0");
    }

    // --- pareto ---
    {
        Generator gen(1111);
        auto x = gen.pareto(3.0, {100});

        test::check(x.min() >= 0.0, "pareto: all non-negative");
    }

    // --- laplace ---
    {
        Generator gen(1212);
        auto x = gen.laplace(0.0, 1.0, {100});

        test::check(x.shape[0] == 100, "laplace: shape");
    }

    // --- triangular ---
    {
        Generator gen(1313);
        auto x = gen.triangular(0.0, 0.5, 1.0, {100});

        test::check(x.min() >= 0.0, "triangular: min >= left");
        test::check(x.max() <= 1.0, "triangular: max <= right");
    }

    // --- Distribution parameter guards (match NumPy domain rules) ---
    {
        Generator gen(999);
        auto must_throw = [&](auto &&fn, const char *what) {
            bool threw = false;
            try
            {
                fn();
            }
            catch (const std::invalid_argument &)
            {
                threw = true;
            }
            test::check(threw, what);
        };

        must_throw([&] { gen.exponential(-1.0); }, "exponential: negative scale throws");
        must_throw([&] { gen.gamma(0.0); }, "gamma: zero shape throws");
        must_throw([&] { gen.gamma(1.0, 0.0); }, "gamma: zero scale throws");
        must_throw([&] { gen.beta(0.0, 1.0); }, "beta: zero a throws");
        must_throw([&] { gen.beta(1.0, -1.0); }, "beta: negative b throws");
        must_throw([&] { gen.chisquare(0.0); }, "chisquare: zero df throws");
        must_throw([&] { gen.f(0.0, 1.0); }, "f: zero dfnum throws");
        must_throw([&] { gen.f(1.0, 0.0); }, "f: zero dfden throws");
        must_throw([&] { gen.standard_t(0.0); }, "standard_t: zero df throws");
        must_throw([&] { gen.weibull(0.0); }, "weibull: zero a throws");
        must_throw([&] { gen.binomial(-1, 0.5); }, "binomial: negative n throws");
        must_throw([&] { gen.binomial(5, -0.1); }, "binomial: p < 0 throws");
        must_throw([&] { gen.binomial(5, 1.1); }, "binomial: p > 1 throws");
        must_throw([&] { gen.negative_binomial(0, 0.5); }, "negative_binomial: zero n throws");
        must_throw([&] { gen.negative_binomial(5, 0.0); }, "negative_binomial: p = 0 throws");
        must_throw([&] { gen.negative_binomial(5, 1.5); }, "negative_binomial: p > 1 throws");
        must_throw([&] { gen.geometric(0.0); }, "geometric: p = 0 throws");
        must_throw([&] { gen.geometric(1.5); }, "geometric: p > 1 throws");
        must_throw([&] { gen.poisson(-1.0); }, "poisson: negative lam throws");
        must_throw([&] { gen.pareto(0.0); }, "pareto: zero a throws");
        must_throw([&] { gen.power(-2.0); }, "power: negative a throws");
        must_throw([&] { gen.rayleigh(-1.0); }, "rayleigh: negative scale throws");
        must_throw([&] { gen.triangular(1.0, 0.0, 2.0); }, "triangular: mode < left throws");
        must_throw([&] { gen.triangular(0.0, 0.0, 0.0); }, "triangular: degenerate throws");
        must_throw([&] { gen.wald(0.0, 1.0); }, "wald: zero mean throws");
        must_throw([&] { gen.wald(1.0, 0.0); }, "wald: zero scale throws");
        must_throw([&] { gen.zipf(1.0); }, "zipf: a = 1 throws");
        must_throw([&] { gen.logseries(0.0); }, "logseries: p = 0 throws");
        must_throw([&] { gen.logseries(1.0); }, "logseries: p = 1 throws");
        must_throw([&] { gen.vonmises(0.0, -1.0); }, "vonmises: negative kappa throws");
        must_throw([&] { gen.noncentral_chisquare(0.0, 1.0); }, "noncentral_chisquare: zero df throws");
        must_throw([&] { gen.noncentral_chisquare(2.0, -1.0); }, "noncentral_chisquare: negative nonc throws");

        // Degenerate-but-valid distributions (verified against NumPy).
        test::check(gen.exponential(0.0, {4}).sum() == 0.0, "exponential(0): all zeros");
        test::check(gen.poisson(0.0, {4}).sum() == 0, "poisson(0): all zeros");
        test::check(gen.rayleigh(0.0, {4}).sum() == 0.0, "rayleigh(0): all zeros");
        test::check(gen.negative_binomial(5, 1.0, {4}).sum() == 0, "negative_binomial(p=1): all zeros");
        test::check(gen.geometric(1.0, {4}).sum() == 4, "geometric(p=1): all ones");
        auto vm = gen.vonmises(1.0, 0.0, {16});
        bool vm_in_range = true;
        for (std::size_t i = 0; i < vm.size(); ++i)
            if (vm.at(i) < 1.0 - std::numbers::pi || vm.at(i) > 1.0 + std::numbers::pi)
                vm_in_range = false;
        test::check(vm_in_range, "vonmises(kappa=0): uniform on circle");
    }

    // --- Module-level convenience functions ---
    {
        // Set seed for reproducibility
        seed_default_rng(42);

        auto x1 = rand<double>({5});
        test::check(x1.shape[0] == 5, "rand: shape");

        auto x2 = randn<double>({5});
        test::check(x2.shape[0] == 5, "randn: shape");

        auto x3 = randint<int>(0, 100, {5});
        test::check(x3.shape[0] == 5, "randint: shape");

        auto arr = asarray(std::vector<int>{1, 2, 3, 4, 5});
        auto perm = permutation(arr);
        test::check(perm.shape[0] == 5, "permutation (module): shape");

        shuffle(arr);
        test::check(arr.shape[0] == 5, "shuffle (module): shape");
    }

    return test::failures() ? 1 : 0;
}
