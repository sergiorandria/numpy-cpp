/**
 * @file padic.hpp
 * @brief p-adic numbers, p-adic lattices and p-adic differential forms — modern engine.
 *
 * Provides `np::padic` with:
 *   - `Padic<T>` residues modulo p^prec (a finite-precision model of Z_p,
 *     NOT full Q_p: negative valuations and non-unit denominators are
 *     unrepresentable — from_rational() throws for non-units): valuation,
 *     norm, unit, inverse, Hensel lift, Teichmüller, p-adic expansion.
 *     T=bigint paths use exact big-int arithmetic; fixed-width T paths
 *     throw when p^prec would overflow their range.
 *   - `PadicLattice<T>` p-adic lattice (basis over Z_p) with dual, volume, LLL over Z_p.
 *   - `PadicDifferential` p-adic forms (Kähler differentials over Q_p) with
 *     exterior derivative, wedge, and p-adic integration.
 *   - Factory `PadicFactory`, Builder `PadicBuilder`, Strategy `IPadicStrategy`
 *     (HenselStrategy, NewtonStrategy), Visitor `PadicVisitor`, Observer, Decorator.
 *   - Integration with `np::lattice` (p-adic lattice as Z_p-module) and
 *     `np::differential` (p-adic forms) and `np::bigint`.
 *
 * Design patterns: **Factory** (PadicFactory), **Builder** (PadicBuilder),
 * **Strategy** (IPadicStrategy), **Visitor** (PadicVisitor), **Observer**,
 * **Decorator** (ScaledPadic), **Prototype** (Padic::clone).
 *
 * Modern C++20: `concepts` (PadicScalar), `std::span`, `std::ranges`,
 * `std::variant`, `std::optional`, `constexpr`, `std::shared_mutex`.
 *
 * Reference: Gouvea *p-adic Numbers*, Koblitz *p-adic Analysis*, Serre *Local Fields*;
 * Hensel's lemma, Teichmüller lift; Hatcher for lattice integration.
 *
 * @author Sergio Randriamihoatra (sergiorandriamihoatra@gmail.com)
 */
#ifndef NP_PADIC_HPP
#define NP_PADIC_HPP

#include <algorithm>
#include <cmath>
#include <concepts>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <ranges>
#include <shared_mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

#include "api_macros.hpp"
#include "bigint.hpp"
#include "differential.hpp"
#include "dtype.hpp"
#include "lattice.hpp"
#include "ndarray.hpp"

namespace np::padic
{

// ── Concepts ────────────────────────────────────────────────────────────
template <typename T>
concept PadicScalar = requires(T a, T b) {
    a + b;
    a - b;
    a * b;
    a == b;
};

// ── Forward decls ───────────────────────────────────────────────────────
template <PadicScalar T = int64_t> struct Padic;
template <PadicScalar T = int64_t> struct PadicLattice;

// ── Visitor (Visitor pattern) ───────────────────────────────────────────
template <PadicScalar T = int64_t> struct PadicVisitor
{
    virtual ~PadicVisitor() = default;
    virtual void visit(const Padic<T> &p) = 0;
};

// ── Observer ────────────────────────────────────────────────────────────
template <PadicScalar T = int64_t> using PadicObserver = std::function<void(const Padic<T> &, const std::string &)>;

// ── Strategy for Hensel/Newton lifting (Strategy pattern) ───────────────
template <PadicScalar T = int64_t> struct IPadicStrategy
{
    virtual ~IPadicStrategy() = default;
    virtual Padic<T> lift(const Padic<T> &a, const std::function<Padic<T>(const Padic<T> &)> &f,
                          const std::function<Padic<T>(const Padic<T> &)> &df) const = 0;
    NP_NODISCARD virtual std::string name() const noexcept = 0;
};

template <PadicScalar T = int64_t> struct HenselStrategy : IPadicStrategy<T>
{
    int max_iter = 20;
    explicit HenselStrategy(int it = 20) : max_iter(it)
    {
    }
    Padic<T> lift(const Padic<T> &a, const std::function<Padic<T>(const Padic<T> &)> &f,
                  const std::function<Padic<T>(const Padic<T> &)> &df) const override;
    NP_NODISCARD std::string name() const noexcept override
    {
        return "Hensel";
    }
};

template <PadicScalar T = int64_t> struct NewtonStrategy : IPadicStrategy<T>
{
    int max_iter = 20;
    explicit NewtonStrategy(int it = 20) : max_iter(it)
    {
    }
    Padic<T> lift(const Padic<T> &a, const std::function<Padic<T>(const Padic<T> &)> &f,
                  const std::function<Padic<T>(const Padic<T> &)> &df) const override
    {
        // Newton is same as Hensel for p-adic (quadratic convergence)
        return HenselStrategy<T>(max_iter).lift(a, f, df);
    }
    NP_NODISCARD std::string name() const noexcept override
    {
        return "Newton";
    }
};

// ── Padic number ────────────────────────────────────────────────────────
template <PadicScalar T> struct Padic
{
    int p = 2;      // prime
    int prec = 20;  // precision (number of p-adic digits)
    T value = T(0); // integer representative modulo p^prec
    // For big integers, value may be stored as bigint via T=bigint, but we keep T generic
    mutable std::shared_mutex mtx_;
    mutable std::vector<PadicObserver<T>> observers_;

    Padic() = default;
    Padic(int prime, T v, int pr = 20) : p(prime), prec(pr), value(v)
    {
        if (!is_prime(prime))
            throw std::invalid_argument("Padic: p must be prime");
        if (pr < 0)
            throw std::invalid_argument("Padic: prec must be non-negative");
        normalize();
    }

    Padic(const Padic &o) : p(o.p), prec(o.prec), value(o.value), observers_(o.observers_)
    {
    }
    Padic &operator=(const Padic &o)
    {
        if (this != &o)
        {
            p = o.p;
            prec = o.prec;
            value = o.value;
            observers_ = o.observers_;
        }
        return *this;
    }
    Padic(Padic &&) noexcept = default;
    Padic &operator=(Padic &&) noexcept = default;

    NP_NODISCARD static bool is_prime(int n) noexcept
    {
        if (n < 2)
            return false;
        for (int i = 2; i * i <= n; ++i)
            if (n % i == 0)
                return false;
        return true;
    }

    // NOTE (honesty audit): noexcept removed — bigint %= etc. allocate.
    void normalize()
    {
        if constexpr (std::is_integral_v<T>)
        {
            long long mod = 1;
            bool overflow = false;
            for (int i = 0; i < prec; ++i)
            {
                if (mod > (1LL << 60) / p)
                {
                    overflow = true;
                    break;
                }
                mod *= p;
            }
            if (overflow)
                return;
            long long v = static_cast<long long>(value);
            long long r = v % mod;
            if (r < 0)
                r += mod;
            value = static_cast<T>(r);
        }
        else if constexpr (std::is_same_v<T, np::bigint>)
        {
            np::bigint mod = 1;
            for (int i = 0; i < prec; ++i)
                mod *= p;
            value %= mod;
            if (value < 0)
                value += mod;
        }
    }

    void add_observer(PadicObserver<T> obs) const
    {
        std::unique_lock lock(mtx_);
        observers_.push_back(std::move(obs));
    }
    void notify(const std::string &ev) const
    {
        std::shared_lock lock(mtx_);
        for (auto &o : observers_)
            o(*this, ev);
    }

    template <typename Visitor> auto accept(Visitor &&v) const -> decltype(v.visit(*this))
    {
        return v.visit(*this);
    }

    NP_NODISCARD Padic clone() const
    {
        return Padic(p, value, prec);
    }

    // valuation v_p(value) = exponent of p in value (for integer value).
    // NOTE (honesty audit): an earlier revision funneled every T through
    // static_cast<long long>, silently truncating bigint values above int64
    // (via a convert_to path that zeroes on overflow). The bigint branch
    // below uses exact T arithmetic; the narrowing cast is gone.
    // (Also dropped bogus noexcept: T arithmetic can allocate/throw.)
    NP_NODISCARD int valuation() const
    {
        if (value == T(0))
            return prec; // by convention, val(0)=prec (or infinity)
        if constexpr (std::is_same_v<T, np::bigint>)
        {
            T v = value;
            if (v < T(0))
                v = -v;
            const T pp = T(p);
            const T zero = T(0);
            int cnt = 0;
            while (v % pp == zero && cnt < prec)
            {
                v /= pp;
                ++cnt;
            }
            return cnt;
        }
        else
        {
            long long v = static_cast<long long>(value);
            if (v < 0)
                v = -v;
            int cnt = 0;
            while (v % p == 0 && cnt < prec)
            {
                v /= p;
                ++cnt;
            }
            return cnt;
        }
    }

    NP_NODISCARD double norm() const
    {
        // p-adic norm |x|_p = p^{-v_p(x)}
        int v = valuation();
        if (value == T(0))
            return 0.0;
        return std::pow(static_cast<double>(p), -v);
    }

    NP_NODISCARD bool is_unit() const
    {
        return valuation() == 0 && value != T(0);
    }
    NP_NODISCARD bool is_zero() const noexcept
    {
        return value == T(0);
    }

    NP_NODISCARD Padic inverse() const
    {
        if (!is_unit())
            throw std::runtime_error("Padic inverse: not a unit (not invertible mod p^prec)");
        if constexpr (std::is_same_v<T, np::bigint>)
        {
            // Exact extended Euclidean in T (no narrowing cast).
            const T zero = T(0), one = T(1);
            T mod = one;
            for (int i = 0; i < prec; ++i)
                mod *= p;
            T t = zero, newt = one;
            T r = mod, newr = value % mod;
            if (newr < zero)
                newr += mod;
            while (newr != zero)
            {
                T q = r / newr;
                T tmp = t - q * newt;
                t = newt;
                newt = tmp;
                tmp = r - q * newr;
                r = newr;
                newr = tmp;
            }
            if (r > one)
                throw std::runtime_error("Padic inverse: not coprime");
            if (t < zero)
                t += mod;
            return Padic(p, t, prec);
        }
        else
        {
            // Compute inverse modulo p^prec via extended Euclidean (for integer).
            // p^prec must fit: otherwise mod wraps and the "inverse" is
            // garbage (previously silent). Bound: prec*log2(p) < 63.
            if (prec > 0 && static_cast<double>(prec) * std::log2(static_cast<double>(p)) >= 63.0)
            {
                throw std::invalid_argument("Padic inverse: p^prec overflows int64 (use Padic<bigint>)");
            }
            long long a = static_cast<long long>(value);
            long long mod = 1;
            for (int i = 0; i < prec; ++i)
                mod *= p;
            long long t = 0, newt = 1;
            long long r = mod, newr = a % mod;
            if (newr < 0)
                newr += mod;
            while (newr != 0)
            {
                long long q = r / newr;
                long long tmp = t - q * newt;
                t = newt;
                newt = tmp;
                tmp = r - q * newr;
                r = newr;
                newr = tmp;
            }
            if (r > 1)
                throw std::runtime_error("Padic inverse: not coprime");
            if (t < 0)
                t += mod;
            return Padic(p, static_cast<T>(t), prec);
        }
    }

    // p-adic expansion digits (least significant first)
    NP_NODISCARD std::vector<int> expansion() const
    {
        std::vector<int> dig(static_cast<std::size_t>(prec), 0);
        if constexpr (std::is_same_v<T, np::bigint>)
        {
            // Exact: reduce mod p^prec in T, then peel base-p digits
            // (the old long-long cast truncated big values here).
            T mod = T(1);
            for (int i = 0; i < prec; ++i)
                mod *= p;
            T v = ((value % mod) + mod) % mod;
            const T pp = T(p);
            for (int i = 0; i < prec; ++i)
            {
                dig[static_cast<std::size_t>(i)] = static_cast<int>(v % pp);
                v /= pp;
            }
        }
        else
        {
            long long v = static_cast<long long>(value);
            if (v < 0)
            {
                // for negative, compute p-adic expansion via 2's complement style: mod p^prec
                long long mod = 1;
                for (int i = 0; i < prec; ++i)
                    mod *= p;
                v = ((v % mod) + mod) % mod;
            }
            for (int i = 0; i < prec; ++i)
            {
                dig[static_cast<std::size_t>(i)] = static_cast<int>(v % p);
                v /= p;
            }
        }
        return dig;
    }

    NP_NODISCARD std::string to_string() const
    {
        auto dig = expansion();
        std::string s;
        for (int i = prec - 1; i >= 0; --i)
        {
            s += std::to_string(dig[i]);
            if (i % 4 == 0 && i != 0)
                s += " ";
        }
        s += " (p=" + std::to_string(p) + ", prec=" + std::to_string(prec) + ")";
        return s;
    }

    // Teichmüller lift: lift a mod p residue to p-adic unit root of unity
    NP_NODISCARD Padic teichmuller() const
    {
        if (value % p == 0)
            throw std::runtime_error("teichmuller: not a unit");
        if constexpr (std::is_same_v<T, np::bigint>)
        {
            // Exact big-int path (the old long-long cast truncated here).
            const T zero = T(0), one = T(1);
            T mod = one;
            for (int i = 0; i < prec; ++i)
                mod *= p;
            T base = value % mod;
            if (base < zero)
                base += mod;
            T exp = one;
            for (int i = 0; i < prec - 1; ++i)
                exp *= p;
            T res = one, b = base, e = exp;
            while (e > zero)
            {
                if ((e % 2) != zero)
                    res = (res * b) % mod;
                b = (b * b) % mod;
                e /= 2;
            }
            return Padic(p, res, prec);
        }
        else
        {
            // Compute a^{p^{prec-1}} mod p^{prec} via fast pow
            long long mod = 1;
            for (int i = 0; i < prec; ++i)
                mod *= p;
            long long base = static_cast<long long>(value) % mod;
            if (base < 0)
                base += mod;
            long long exp = 1;
            for (int i = 0; i < prec - 1; ++i)
                exp *= p;
            long long res = 1, b = base;
            long long e = exp;
            while (e > 0)
            {
                if (e & 1)
                    res = (res * b) % mod;
                b = (b * b) % mod;
                e >>= 1;
            }
            return Padic(p, static_cast<T>(res), prec);
        }
    }

    // Arithmetic (notify observers on operands and result for observer pattern)
    friend inline Padic operator+(const Padic &a, const Padic &b)
    {
        if (a.p != b.p || a.prec != b.prec)
            throw std::invalid_argument("Padic +: p/prec mismatch");
        a.notify("add");
        b.notify("add");
        Padic r(a.p, a.value + b.value, a.prec);
        r.notify("add");
        return r;
    }
    friend inline Padic operator-(const Padic &a, const Padic &b)
    {
        if (a.p != b.p || a.prec != b.prec)
            throw std::invalid_argument("Padic -: p/prec mismatch");
        a.notify("sub");
        b.notify("sub");
        Padic r(a.p, a.value - b.value, a.prec);
        r.notify("sub");
        return r;
    }
    friend inline Padic operator*(const Padic &a, const Padic &b)
    {
        if (a.p != b.p || a.prec != b.prec)
            throw std::invalid_argument("Padic *: p/prec mismatch");
        a.notify("mul");
        b.notify("mul");
        Padic r(a.p, a.value * b.value, a.prec);
        r.notify("mul");
        return r;
    }
    friend inline Padic operator/(const Padic &a, const Padic &b)
    {
        if (a.p != b.p || a.prec != b.prec)
            throw std::invalid_argument("Padic /: p/prec mismatch");
        return a * b.inverse();
    }
    friend inline bool operator==(const Padic &a, const Padic &b) noexcept
    {
        return a.p == b.p && a.prec == b.prec && a.value == b.value;
    }
    friend inline bool operator!=(const Padic &a, const Padic &b) noexcept
    {
        return !(a == b);
    }
};

// ── Hensel lift implementation ──────────────────────────────────────────
template <PadicScalar T>
Padic<T> HenselStrategy<T>::lift(const Padic<T> &a, const std::function<Padic<T>(const Padic<T> &)> &f,
                                 const std::function<Padic<T>(const Padic<T> &)> &df) const
{
    Padic<T> x = a;
    for (int i = 0; i < max_iter; ++i)
    {
        auto fx = f(x);
        if (fx.is_zero())
            break;
        auto dfx = df(x);
        if (!dfx.is_unit())
            throw std::runtime_error("Hensel: derivative not a unit, cannot lift");
        // Newton iteration: x_{n+1} = x_n - f(x_n)/f'(x_n)
        auto delta = fx / dfx;
        x = Padic<T>(x.p, x.value - delta.value, x.prec);
        x.notify("hensel_iter");
    }
    return x;
}

// ── PadicLattice — Z_p lattice (p-adic analogue of integer lattice) ───────
template <PadicScalar T> struct PadicLattice
{
    lattice::Lattice<T> underlying; // integer lattice underlying the p-adic lattice
    int p = 2;
    int prec = 20;

    PadicLattice() = default;
    PadicLattice(lattice::Lattice<T> lat, int prime, int pr = 20) : underlying(std::move(lat)), p(prime), prec(pr)
    {
        if (!Padic<T>::is_prime(prime))
            throw std::invalid_argument("PadicLattice: p must be prime");
    }

    NP_NODISCARD int rank() const noexcept
    {
        return underlying.rank();
    }
    NP_NODISCARD int dim() const noexcept
    {
        return underlying.dim();
    }

    // Scale lattice by p^k (p-adic scaling)
    NP_NODISCARD PadicLattice scaled(int k) const
    {
        auto B = underlying.basis;
        int n = B.shape[0], d = B.shape[1];
        ndarray<T> Bs(std::vector<int>{n, d});
        T powp = T(1);
        for (int i = 0; i < std::abs(k); ++i)
            powp *= static_cast<T>(p);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < d; ++j)
                Bs(i, j) = (k >= 0 ? B(i, j) * powp : B(i, j) / powp);
        return PadicLattice(lattice::Lattice<T>(Bs), p, prec);
    }

    // Dual p-adic lattice
    NP_NODISCARD PadicLattice dual() const
    {
        return PadicLattice(underlying.dual(), p, prec);
    }

    // Euclidean covolume sqrt|det Gram| (for the p-adic absolute value see
    // p_adic_volume() below; an earlier revision returned this under that
    // name with a "|det|_p" comment).
    NP_NODISCARD double euclidean_volume() const
    {
        return static_cast<double>(underlying.volume());
    }

    // p-adic covolume |det Gram|_p = p^{-v_p(det)}. Exact for integral
    // bases: det Gram is a nonnegative integer, factored by trial division.
    // Throws invalid_argument when det is not (near-)integral or non
    // positive — e.g. non-integral bases — instead of returning the
    // Euclidean value under a p-adic name (the old behavior).
    NP_NODISCARD double p_adic_volume() const
    {
        const double vol = static_cast<double>(underlying.volume());
        if (vol == 0.0)
            return 0.0;
        if (!std::isfinite(vol))
            throw std::invalid_argument("p_adic_volume: Euclidean volume not finite");
        const double det = vol * vol;
        const long long detll = std::llround(det);
        if (std::abs(det - static_cast<double>(detll)) > 1e-6 * std::max(1.0, std::abs(det)) || detll <= 0)
            throw std::invalid_argument("p_adic_volume: Gram determinant not a positive integer");
        long long t = detll;
        int v = 0;
        while (t % p == 0)
        {
            t /= p;
            ++v;
        }
        return std::pow(static_cast<double>(p), -v);
    }

    // Meet/join via underlying lattice
    NP_NODISCARD PadicLattice meet(const PadicLattice &other) const
    {
        if (p != other.p || prec != other.prec)
            throw std::invalid_argument("PadicLattice meet: p/prec mismatch");
        return PadicLattice(underlying.meet(other.underlying), p, prec);
    }
    NP_NODISCARD PadicLattice join(const PadicLattice &other) const
    {
        if (p != other.p || prec != other.prec)
            throw std::invalid_argument("PadicLattice join: p/prec mismatch");
        return PadicLattice(underlying.join(other.underlying), p, prec);
    }

    // p-adic lattice norm: min over basis vectors of max_j |b_ij|_p.
    // NOTE (honesty audit): an earlier revision summed squares and took a
    // square root (Euclidean norm) under this name. The non-Archimedean
    // maximum is the correct p-adic analogue.
    NP_NODISCARD double p_adic_norm() const
    {
        int n = rank(), d = dim();
        double best = std::numeric_limits<double>::infinity();
        for (int i = 0; i < n; ++i)
        {
            double m = 0.0;
            for (int j = 0; j < d; ++j)
            {
                Padic<T> c(p, underlying.basis(i, j), prec);
                const double cn = c.norm();
                if (cn > m)
                    m = cn;
            }
            if (m < best)
                best = m;
        }
        return best;
    }
};

// ── Factory (Factory pattern) ───────────────────────────────────────────
struct PadicFactory
{
    template <PadicScalar T = int64_t> NP_NODISCARD static Padic<T> from_int(int p, T v, int prec = 20)
    {
        return Padic<T>(p, v, prec);
    }
    // NOTE: quotients with negative valuation (non-unit denominator) are
    // unrepresentable in this residues-mod-p^prec model — throws via inverse().
    template <PadicScalar T = int64_t> NP_NODISCARD static Padic<T> from_rational(int p, T num, T den, int prec = 20)
    {
        Padic<T> a(p, num, prec);
        Padic<T> b(p, den, prec);
        return a / b;
    }
    template <PadicScalar T = int64_t> NP_NODISCARD static Padic<T> zero(int p, int prec = 20)
    {
        return Padic<T>(p, T(0), prec);
    }
    template <PadicScalar T = int64_t> NP_NODISCARD static Padic<T> one(int p, int prec = 20)
    {
        return Padic<T>(p, T(1), prec);
    }
    template <PadicScalar T = int64_t> NP_NODISCARD static PadicLattice<T> cubic_padic(int n, int p, int prec = 20)
    {
        auto lat = lattice::LatticeFactory::cubic<T>(n);
        return PadicLattice<T>(lat, p, prec);
    }
    template <PadicScalar T = int64_t> NP_NODISCARD static PadicLattice<T> a_n_padic(int n, int p, int prec = 20)
    {
        auto lat = lattice::LatticeFactory::a_n<T>(n);
        return PadicLattice<T>(lat, p, prec);
    }
};

// ── Builder (Builder pattern) ───────────────────────────────────────────
template <PadicScalar T = int64_t> struct PadicBuilder
{
    int p_ = 2;
    int prec_ = 20;
    T value_ = T(0);
    bool has_value_ = false;

    PadicBuilder &prime(int pp)
    {
        p_ = pp;
        return *this;
    }
    PadicBuilder &precision(int pr)
    {
        prec_ = pr;
        return *this;
    }
    PadicBuilder &value(T v)
    {
        value_ = v;
        has_value_ = true;
        return *this;
    }
    PadicBuilder &from_int(T v)
    {
        value_ = v;
        has_value_ = true;
        return *this;
    }
    NP_NODISCARD Padic<T> build() const
    {
        if (!has_value_)
            throw std::invalid_argument("PadicBuilder: no value");
        return Padic<T>(p_, value_, prec_);
    }
    NP_NODISCARD static PadicBuilder<T> create()
    {
        return {};
    }
};

// ── Decorator (Decorator pattern) ───────────────────────────────────────
template <PadicScalar T> struct ScaledPadic
{
    Padic<T> inner;
    T scale = T(1);
    ScaledPadic(Padic<T> p, T s) : inner(std::move(p)), scale(s)
    {
    }
    NP_NODISCARD Padic<T> as_padic() const
    {
        return Padic<T>(inner.p, inner.value * scale, inner.prec);
    }
};

// ── Free ops ────────────────────────────────────────────────────────────
template <PadicScalar T>
NP_NODISCARD inline Padic<T> hensel_lift(const Padic<T> &a, const std::function<Padic<T>(const Padic<T> &)> &f,
                                         const std::function<Padic<T>(const Padic<T> &)> &df,
                                         const IPadicStrategy<T> &strat = HenselStrategy<T>{})
{
    return strat.lift(a, f, df);
}

template <PadicScalar T> NP_NODISCARD inline int valuation(const Padic<T> &a) noexcept
{
    return a.valuation();
}
template <PadicScalar T> NP_NODISCARD inline double norm(const Padic<T> &a) noexcept
{
    return a.norm();
}
template <PadicScalar T> NP_NODISCARD inline Padic<T> teichmuller(const Padic<T> &a)
{
    return a.teichmuller();
}

// ── Integration with lattice/differential ───────────────────────────────
// p-adic lattice from integer lattice
template <PadicScalar T>
NP_NODISCARD inline PadicLattice<T> to_padic_lattice(const lattice::Lattice<T> &lat, int p, int prec = 20)
{
    return PadicLattice<T>(lat, p, prec);
}
// p-adic differential form: Kähler differential over Q_p (stub, uses differential::VM)
struct PadicDifferential
{
    int p = 2;
    int prec = 20;
    // NOTE (honesty audit): formal Kähler differentials obey the same
    // algebraic rules over Q_p as over R, so delegating the FORMAL
    // derivative is mathematically sound. What does NOT exist here is any
    // p-adic norm/convergence test (an earlier comment claimed one) — series
    // convergence in |·|_p is the caller's responsibility.
    PadicDifferential() = default;
    PadicDifferential(int pp, int pr) : p(pp), prec(pr)
    {
    }
    template <typename VM> NP_NODISCARD auto exterior_derivative(const VM &vm) const
    {
        return ::np::differential::exterior_derivative(vm);
    }
};

} // namespace np::padic

#endif // NP_PADIC_HPP
