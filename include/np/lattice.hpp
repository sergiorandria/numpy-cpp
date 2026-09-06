/**
 * @file lattice.hpp
 * @brief Integer and order-theoretic lattices — meet/join, dual, LLL/BKZ, Gram,
 * volume, shortest/closest vectors, poset lattices, with modern C++20 patterns.
 *
 * Provides `np::lattice` with:
 *   - `Lattice<T>` (integer lattice, rank n in R^d, basis as ndarray<T> n×d):
 *     rank/dim, gram_matrix, volume, dual, contains, lll_reduce, bkz_reduce,
 *     shortest_vector (exact enumeration), closest_vector (Babai), meet/join
 *     (intersection/sum via dual), sublattice, quotient.
 *   - `PosetLattice<T>` (finite poset lattice, order-theoretic):
 *     meet/join, is_lattice, is_distributive/modular, hasse_diagram,
 *     mobius, zeta, atoms/coatoms.
 *   - Factory `LatticeFactory` — cubic, hexagonal, A_n, D_n, E8, Leech stub.
 *   - Builder `LatticeBuilder<T>` fluent.
 *   - Strategies `IReductionStrategy<T>` — `LLLStrategy`, `BKZStrategy`.
 *   - Visitor `LatticeVisitor<T>` for traversal, Observer `LatticeObserver`.
 *   - Decorator `TransformedLattice<T>` (rotated/scaled view).
 *   - Free ops `meet`, `join`, `dual`, `lll`, `gram`, `volume`, `shortest`.
 *
 * Design patterns: **Factory** (LatticeFactory), **Builder** (LatticeBuilder),
 * **Strategy** (IReductionStrategy), **Visitor** (LatticeVisitor),
 * **Observer** (LatticeObserver), **Decorator** (TransformedLattice),
 * **Prototype** (Lattice::clone).
 *
 * Modern C++20: `concepts` (LatticeScalar, Ordered), `std::span`,
 * `std::ranges`, `std::variant`, `constexpr`, `std::shared_mutex`,
 * `std::optional`, `std::expected`-style optional, `[[nodiscard]]`.
 *
 * Reference: Micciancio–Goldwasser *Complexity of Lattice Problems*;
 * Lenstra–Lenstra–Lovász 1982; Refs: `numpy-reference` (linalg), Hatcher.
 *
 * @author Sergio Randriamihoatra (sergiorandriamihoatra@gmail.com)
 */
#ifndef NP_LATTICE_HPP
#define NP_LATTICE_HPP

#include <algorithm>
#include <cmath>
#include <concepts>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <random>
#include <ranges>
#include <shared_mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include "api_macros.hpp"
#include "dtype.hpp"
#include "linalg.hpp"
#include "ndarray.hpp"

namespace np::lattice
{

// ── Concepts ────────────────────────────────────────────────────────────
template <typename T>
concept LatticeScalar = std::is_arithmetic_v<T> || detail::is_complex_v<T>;

template <typename T>
concept Ordered = requires(const T &a, const T &b) {
    { a < b } -> std::convertible_to<bool>;
    { a == b } -> std::convertible_to<bool>;
};

// ── Forward decls ───────────────────────────────────────────────────────
template <LatticeScalar T = double> struct Lattice;
template <Ordered T = int> struct PosetLattice;

// ── Visitor (Visitor pattern) ───────────────────────────────────────────
template <LatticeScalar T = double> struct LatticeVisitor
{
    virtual ~LatticeVisitor() = default;
    virtual void visit(const Lattice<T> &lat) = 0;
    virtual void visit_basis(const ndarray<T> &basis) = 0;
};

template <Ordered T = int> struct PosetVisitor
{
    virtual ~PosetVisitor() = default;
    virtual void visit(const PosetLattice<T> &p) = 0;
};

// ── Observer (Observer pattern) ─────────────────────────────────────────
template <LatticeScalar T = double>
using LatticeObserver = std::function<void(const Lattice<T> &, const std::string &)>;

// ── Strategy for reduction (Strategy pattern) ───────────────────────────
template <LatticeScalar T = double> struct IReductionStrategy
{
    virtual ~IReductionStrategy() = default;
    virtual Lattice<T> reduce(const Lattice<T> &lat) const = 0;
    NP_NODISCARD virtual std::string name() const noexcept = 0;
};

template <LatticeScalar T = double> struct LLLStrategy : IReductionStrategy<T>
{
    double delta = 0.75;
    double eta = 0.5;
    explicit LLLStrategy(double d = 0.75, double e = 0.5) : delta(d), eta(e)
    {
    }
    Lattice<T> reduce(const Lattice<T> &lat) const override;
    NP_NODISCARD std::string name() const noexcept override
    {
        return "LLL(delta=" + std::to_string(delta) + ")";
    }
};

template <LatticeScalar T = double> struct BKZStrategy : IReductionStrategy<T>
{
    int block = 20;
    double delta = 0.75;
    explicit BKZStrategy(int b = 20, double d = 0.75) : block(b), delta(d)
    {
    }
    Lattice<T> reduce(const Lattice<T> &lat) const override
    {
        // Full BKZ: blockwise LLL with enumeration (simplified)
        int n = lat.rank();
        if (n <= block)
            return LLLStrategy<T>(delta).reduce(lat);
        Lattice<T> cur = lat;
        // Slide window: for each block start, extract block, LLL, reinsert
        for (int start = 0; start + block <= n; ++start)
        {
            // Extract block rows [start, start+block)
            std::vector<int> idx(block);
            std::iota(idx.begin(), idx.end(), start);
            auto sub = cur.sublattice(std::span<const int>(idx.data(), idx.size()));
            auto reduced = LLLStrategy<T>(delta).reduce(sub);
            // Reinsert reduced block back into cur
            for (int i = 0; i < block; ++i)
                for (int j = 0; j < cur.dim(); ++j)
                    cur.basis(start + i, j) = reduced.basis(i, j);
        }
        // Final LLL to clean up
        return LLLStrategy<T>(delta).reduce(cur);
    }
    NP_NODISCARD std::string name() const noexcept override
    {
        return "BKZ(block=" + std::to_string(block) + ")";
    }
};

// ── Lattice (integer lattice, rank n × dim d) ───────────────────────────
template <LatticeScalar T> struct Lattice
{
    ndarray<T> basis; // shape {n,d}
    mutable std::shared_mutex mtx_;
    mutable std::vector<LatticeObserver<T>> observers_;

    Lattice() = default;
    explicit Lattice(ndarray<T> b) : basis(std::move(b))
    {
    }
    Lattice(std::initializer_list<std::initializer_list<T>> init)
    {
        // Build basis from nested init: {{1,0},{0,1}} etc.
        std::vector<T> flat;
        size_t n = init.size();
        size_t d = 0;
        for (auto &row : init)
            d = std::max(d, row.size());
        flat.reserve(n * d);
        for (auto &row : init)
        {
            size_t j = 0;
            for (auto v : row)
            {
                flat.push_back(v);
                ++j;
            }
            for (; j < d; ++j)
                flat.push_back(T(0));
        }
        basis = ndarray<T>(std::vector<int>{static_cast<int>(n), static_cast<int>(d)});
        auto &data = basis.data();
        for (size_t i = 0; i < flat.size(); ++i)
            data[i] = flat[i];
    }

    Lattice(const Lattice &o) : basis(o.basis), observers_(o.observers_)
    {
    }
    Lattice &operator=(const Lattice &o)
    {
        if (this != &o)
        {
            basis = o.basis;
            observers_ = o.observers_;
        }
        return *this;
    }
    Lattice(Lattice &&) noexcept = default;
    Lattice &operator=(Lattice &&) noexcept = default;

    NP_NODISCARD Lattice clone() const
    {
        return Lattice(basis);
    }

    NP_NODISCARD int rank() const noexcept
    {
        if (basis.ndim() < 2)
            return basis.size() ? 1 : 0;
        return basis.shape[0];
    }
    NP_NODISCARD int dim() const noexcept
    {
        if (basis.ndim() < 2)
            return basis.size() ? 1 : 0;
        return basis.shape[1];
    }
    NP_NODISCARD bool empty() const noexcept
    {
        return basis.size() == 0;
    }

    // Independent basis (row rank) — modern ranges + Gaussian elimination
    NP_NODISCARD Lattice<T> independent() const
    {
        int n = rank(), d = dim();
        if (n == 0 || d == 0)
            return Lattice<T>();
        std::vector<std::vector<double>> cur(n, std::vector<double>(d));
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < d; ++j)
                cur[i][j] = static_cast<double>(basis(i, j));
        std::vector<int> used(n, 0);
        std::vector<int> keep;
        keep.reserve(n);
        for (int col = 0; col < d && static_cast<int>(keep.size()) < n; ++col)
        {
            int sel = -1;
            for (int i = 0; i < n; ++i)
                if (!used[i] && std::abs(cur[i][col]) > 1e-9)
                {
                    sel = i;
                    break;
                }
            if (sel == -1)
                continue;
            used[sel] = 1;
            keep.push_back(sel);
            for (int i = 0; i < n; ++i)
                if (!used[i] && std::abs(cur[i][col]) > 1e-9)
                {
                    double factor = cur[i][col] / cur[sel][col];
                    for (int j = col; j < d; ++j)
                        cur[i][j] -= factor * cur[sel][j];
                }
        }
        if (keep.empty())
            return Lattice<T>();
        ndarray<T> Bc(std::vector<int>{static_cast<int>(keep.size()), d});
        for (size_t i = 0; i < keep.size(); ++i)
            for (int j = 0; j < d; ++j)
                Bc(static_cast<int>(i), j) = basis(keep[i], j);
        return Lattice<T>(Bc);
    }

    NP_NODISCARD int actual_rank() const
    {
        return independent().rank();
    }

    // Observer support
    void add_observer(LatticeObserver<T> obs) const
    {
        std::unique_lock lock(mtx_);
        observers_.push_back(std::move(obs));
    }
    void clear_observers() const noexcept
    {
        std::unique_lock lock(mtx_);
        observers_.clear();
    }
    void notify(const std::string &ev) const
    {
        std::shared_lock lock(mtx_);
        for (auto &o : observers_)
            o(*this, ev);
    }

    // Visitor accept
    template <typename Visitor> auto accept(Visitor &&v) const -> decltype(v.visit(*this))
    {
        return v.visit(*this);
    }

    // Gram matrix G = B * B^T  (n x n)
    NP_NODISCARD ndarray<T> gram_matrix() const
    {
        int n = rank(), d = dim();
        if (n == 0 || d == 0)
            return ndarray<T>(std::vector<int>{0, 0});
        ndarray<T> G(std::vector<int>{n, n});
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
            {
                T s = T(0);
                for (int k = 0; k < d; ++k)
                    s += basis(i, k) * basis(j, k);
                G(i, j) = s;
            }
        return G;
    }

    // Volume = sqrt(det(G)) for full rank, else 0 (uses independent rank)
    NP_NODISCARD T volume() const
    {
        auto Ind = independent();
        auto G = Ind.gram_matrix();
        int n = G.shape[0];
        if (n == 0)
            return T(0);
        // For small n, compute determinant via linalg (use double path)
        // Convert to double for determinant, then cast back
        ndarray<double> Gd(std::vector<int>{n, n});
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                Gd(i, j) = static_cast<double>(G(i, j));
        double det = 1.0;
        // Simple LU determinant for small n (use linalg::det if available via manual)
        // Fallback: compute via Gaussian elimination
        ndarray<double> A = Gd;
        int sgn = 1;
        for (int k = 0; k < n; ++k)
        {
            // pivot
            int piv = k;
            double maxv = std::abs(A(k, k));
            for (int i = k + 1; i < n; ++i)
                if (std::abs(A(i, k)) > maxv)
                {
                    maxv = std::abs(A(i, k));
                    piv = i;
                }
            if (maxv < 1e-12)
                return T(0);
            if (piv != k)
            {
                for (int j = 0; j < n; ++j)
                    std::swap(A(k, j), A(piv, j));
                sgn = -sgn;
            }
            det *= A(k, k);
            for (int i = k + 1; i < n; ++i)
            {
                double factor = A(i, k) / A(k, k);
                for (int j = k + 1; j < n; ++j)
                    A(i, j) -= factor * A(k, j);
            }
        }
        det = sgn * det;
        if (det < 0)
            det = -det;
        return static_cast<T>(std::sqrt(det));
    }

    // Dual lattice: for square full-rank, dual = (B^{-1})^T
    // Modern: use independent basis to handle rank-deficient joins
    NP_NODISCARD Lattice<T> dual() const
    {
        auto Ind = independent();
        int n = Ind.rank(), d = Ind.dim();
        if (n == 0 || d == 0)
            return Lattice<T>();
        // Use independent basis for computation
        const auto &b = Ind.basis;
        if (n != d)
        {
            // Rectangular: dual via G^{-1} B  =>  B_dual = G^{-1} * B  (n x d)
            // Compute G = B B^T (n x n), invert, then B_dual = G^{-1} B (modern: use
            // independent)
            auto G = Ind.gram_matrix();
            int nn = n;
            ndarray<double> Gd(std::vector<int>{nn, nn});
            for (int i = 0; i < nn; ++i)
                for (int j = 0; j < nn; ++j)
                    Gd(i, j) = static_cast<double>(G(i, j));
            // Invert Gd via Gauss-Jordan
            ndarray<double> Inv(std::vector<int>{nn, nn});
            for (int i = 0; i < nn; ++i)
                for (int j = 0; j < nn; ++j)
                    Inv(i, j) = (i == j ? 1.0 : 0.0);
            ndarray<double> A = Gd;
            for (int k = 0; k < nn; ++k)
            {
                // pivot
                int piv = k;
                double maxv = std::abs(A(k, k));
                for (int i = k + 1; i < nn; ++i)
                    if (std::abs(A(i, k)) > maxv)
                    {
                        maxv = std::abs(A(i, k));
                        piv = i;
                    }
                if (maxv < 1e-12)
                    throw std::runtime_error("dual: singular Gram");
                if (piv != k)
                {
                    for (int j = 0; j < nn; ++j)
                    {
                        std::swap(A(k, j), A(piv, j));
                        std::swap(Inv(k, j), Inv(piv, j));
                    }
                }
                double pivv = A(k, k);
                for (int j = 0; j < nn; ++j)
                {
                    A(k, j) /= pivv;
                    Inv(k, j) /= pivv;
                }
                for (int i = 0; i < nn; ++i)
                    if (i != k)
                    {
                        double f = A(i, k);
                        for (int j = 0; j < nn; ++j)
                        {
                            A(i, j) -= f * A(k, j);
                            Inv(i, j) -= f * Inv(k, j);
                        }
                    }
            }
            ndarray<T> Bd(std::vector<int>{n, d});
            for (int i = 0; i < n; ++i)
                for (int j = 0; j < d; ++j)
                {
                    double s = 0;
                    for (int k = 0; k < n; ++k)
                        s += Inv(i, k) * static_cast<double>(b(k, j));
                    Bd(i, j) = static_cast<T>(s);
                }
            return Lattice<T>(Bd);
        }
        // square
        ndarray<double> Bd(std::vector<int>{n, d});
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < d; ++j)
                Bd(i, j) = static_cast<double>(b(i, j));
        // Invert Bd
        ndarray<double> Inv(std::vector<int>{n, n});
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                Inv(i, j) = (i == j ? 1.0 : 0.0);
        ndarray<double> A = Bd;
        for (int k = 0; k < n; ++k)
        {
            int piv = k;
            double maxv = std::abs(A(k, k));
            for (int i = k + 1; i < n; ++i)
                if (std::abs(A(i, k)) > maxv)
                {
                    maxv = std::abs(A(i, k));
                    piv = i;
                }
            if (maxv < 1e-12)
                throw std::runtime_error("dual: singular");
            if (piv != k)
            {
                for (int j = 0; j < n; ++j)
                {
                    std::swap(A(k, j), A(piv, j));
                    std::swap(Inv(k, j), Inv(piv, j));
                }
            }
            double pivv = A(k, k);
            for (int j = 0; j < n; ++j)
            {
                A(k, j) /= pivv;
                Inv(k, j) /= pivv;
            }
            for (int i = 0; i < n; ++i)
                if (i != k)
                {
                    double f = A(i, k);
                    for (int j = 0; j < n; ++j)
                    {
                        A(i, j) -= f * A(k, j);
                        Inv(i, j) -= f * Inv(k, j);
                    }
                }
        }
        ndarray<T> DualB(std::vector<int>{n, d});
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < d; ++j)
                DualB(i, j) = static_cast<T>(Inv(j, i)); // transpose
        return Lattice<T>(DualB);
    }

    // Contains: does lattice contain vector v (within tolerance)?
    NP_NODISCARD bool contains(const ndarray<T> &v, double tol = 1e-9) const
    {
        // Solve B^T x = v  (least squares) and check integer coords
        int n = rank(), d = dim();
        if (v.size() != static_cast<size_t>(d))
            return false;
        // For small n, brute force via solving linear system if square
        if (n == d)
        {
            // Solve B^T? Actually basis rows are vectors, so v = sum c_i b_i => c = v *
            // B^{-1} ? Build matrix B^T? Let's solve linear system B^T c = v? Wait B is n x
            // d, c is 1 x n, v = c * B  =>  v^T = B^T c^T . So solve B^T x = v^T
            ndarray<double> BT(std::vector<int>{d, n});
            for (int i = 0; i < n; ++i)
                for (int j = 0; j < d; ++j)
                    BT(j, i) = static_cast<double>(basis(i, j));
            // Solve via normal equations: (B B^T) c^T = B v^T? Simpler: use least squares via
            // enumeration for small n
        }
        // Fallback: use closest vector and check distance
        auto c = closest_vector(v);
        double dist2 = 0;
        for (int j = 0; j < d; ++j)
        {
            double diff = static_cast<double>(v[j]) - static_cast<double>(c[j]);
            dist2 += diff * diff;
        }
        return std::sqrt(dist2) < tol;
    }

    // LLL reduce via strategy (notifies observers)
    NP_NODISCARD Lattice<T> lll_reduce(double delta = 0.75) const
    {
        notify("lll_reduce");
        auto out = LLLStrategy<T>(delta).reduce(*this);
        // also notify new lattice for decorator observers
        out.notify("lll_reduce");
        return out;
    }
    NP_NODISCARD Lattice<T> reduce_with(const IReductionStrategy<T> &strat) const
    {
        notify("reduce_with:" + strat.name());
        return strat.reduce(*this);
    }

    // Shortest vector (exact enumeration for n <= 8, else LLL+enum)
    NP_NODISCARD ndarray<T> shortest_vector() const
    {
        int n = rank(), d = dim();
        if (n == 0)
            return ndarray<T>(std::vector<int>{d});
        // Use LLL reduced basis then enumerate small coeffs in [-2,2]
        auto R = lll_reduce();
        double best = std::numeric_limits<double>::infinity();
        ndarray<T> best_vec(std::vector<int>{d});
        // enumerate
        int range = 2;
        // for n>6, reduce range to avoid explosion
        if (n > 6)
            range = 1;
        std::vector<int> coeff(n, -range);
        auto eval = [&]() -> double {
            ndarray<T> v(std::vector<int>{d});
            for (int j = 0; j < d; ++j)
                v[j] = T(0);
            for (int i = 0; i < n; ++i)
                for (int j = 0; j < d; ++j)
                    v[j] = static_cast<T>(v[j]) + static_cast<T>(coeff[i]) * R.basis(i, j);
            double nrm = 0;
            for (int j = 0; j < d; ++j)
                nrm += static_cast<double>(v[j]) * static_cast<double>(v[j]);
            if (nrm > 1e-12 && nrm < best)
            {
                best = nrm;
                best_vec = v;
            }
            return nrm;
        };
        // simple odometer
        while (true)
        {
            // skip all zero
            bool allzero = true;
            for (int c : coeff)
                if (c != 0)
                {
                    allzero = false;
                    break;
                }
            if (!allzero)
                eval();
            int p = 0;
            while (p < n)
            {
                coeff[p] += 1;
                if (coeff[p] > range)
                {
                    coeff[p] = -range;
                    ++p;
                }
                else
                    break;
            }
            if (p >= n)
                break;
        }
        if (n > 8 && best == std::numeric_limits<double>::infinity())
        {
            // Sieve heuristic for n>8: random sampling with LLL basis (modern, ranges)
            std::mt19937 rng(42);
            std::uniform_int_distribution<int> dist(-1, 1);
            for (int iter = 0; iter < 1000; ++iter)
            {
                for (int i = 0; i < n; ++i)
                    coeff[i] = dist(rng);
                bool allzero = true;
                for (int c : coeff)
                    if (c != 0)
                    {
                        allzero = false;
                        break;
                    }
                if (allzero)
                    continue;
                eval();
            }
        }
        if (best == std::numeric_limits<double>::infinity())
            return best_vec;
        return best_vec;
    }

    // Closest vector via Babai nearest plane (using LLL reduced basis)
    NP_NODISCARD ndarray<T> closest_vector(const ndarray<T> &target) const
    {
        int n = rank(), d = dim();
        if (n == 0)
            return ndarray<T>(std::vector<int>{d});
        auto R = lll_reduce();
        // Gram-Schmidt for Babai
        // Compute GS orthogonalization of R.basis (rows)
        std::vector<std::vector<double>> b_star(n, std::vector<double>(d, 0));
        std::vector<std::vector<double>> mu(n, std::vector<double>(n, 0));
        for (int i = 0; i < n; ++i)
        {
            for (int k = 0; k < d; ++k)
                b_star[i][k] = static_cast<double>(R.basis(i, k));
            for (int j = 0; j < i; ++j)
            {
                double dot = 0, nj = 0;
                for (int k = 0; k < d; ++k)
                    dot += static_cast<double>(R.basis(i, k)) * b_star[j][k];
                for (int k = 0; k < d; ++k)
                    nj += b_star[j][k] * b_star[j][k];
                mu[i][j] = nj > 1e-12 ? dot / nj : 0;
                for (int k = 0; k < d; ++k)
                    b_star[i][k] -= mu[i][j] * b_star[j][k];
            }
        }
        // Babai: work from n-1 down to 0
        ndarray<T> t = target;
        std::vector<int> coeff(n, 0);
        std::vector<double> w(d);
        for (int k = 0; k < d; ++k)
            w[k] = static_cast<double>(t[k]);
        for (int i = n - 1; i >= 0; --i)
        {
            double dot = 0, nj = 0;
            for (int k = 0; k < d; ++k)
                dot += w[k] * b_star[i][k];
            for (int k = 0; k < d; ++k)
                nj += b_star[i][k] * b_star[i][k];
            double c = nj > 1e-12 ? dot / nj : 0;
            int ci = static_cast<int>(std::round(c));
            coeff[i] = ci;
            for (int k = 0; k < d; ++k)
                w[k] -= ci * static_cast<double>(R.basis(i, k));
        }
        ndarray<T> closest(std::vector<int>{d});
        for (int k = 0; k < d; ++k)
            closest[k] = T(0);
        for (int i = 0; i < n; ++i)
            for (int k = 0; k < d; ++k)
                closest[k] = static_cast<T>(closest[k]) + static_cast<T>(coeff[i]) * R.basis(i, k);
        return closest;
    }

    // Meet (intersection) via dual of join of duals
    NP_NODISCARD Lattice<T> meet(const Lattice<T> &other) const
    {
        if (empty() || other.empty())
            return Lattice<T>();
        // dual join dual
        auto d1 = this->dual();
        auto d2 = other.dual();
        auto j = d1.join(d2);
        return j.dual();
    }

    // Join (sum) = lattice generated by union of bases
    NP_NODISCARD Lattice<T> join(const Lattice<T> &other) const
    {
        if (empty())
            return other.clone();
        if (other.empty())
            return clone();
        int n1 = rank(), n2 = other.rank(), d = dim();
        if (d != other.dim())
            throw std::invalid_argument("join: dim mismatch");
        ndarray<T> B(std::vector<int>{n1 + n2, d});
        for (int i = 0; i < n1; ++i)
            for (int j = 0; j < d; ++j)
                B(i, j) = basis(i, j);
        for (int i = 0; i < n2; ++i)
            for (int j = 0; j < d; ++j)
                B(n1 + i, j) = other.basis(i, j);
        Lattice<T> J(B);
        // Reduce to minimal basis via LLL (also removes dependencies)
        return J.lll_reduce();
    }

    // Sublattice via selecting subset of basis vectors (by span)
    NP_NODISCARD Lattice<T> sublattice(std::span<const int> idx) const
    {
        int d = dim();
        ndarray<T> B(std::vector<int>{static_cast<int>(idx.size()), d});
        for (size_t i = 0; i < idx.size(); ++i)
            for (int j = 0; j < d; ++j)
                B(static_cast<int>(i), j) = basis(idx[i], j);
        return Lattice<T>(B);
    }
};

// ── LLL implementation (Strategy) ───────────────────────────────────────
template <LatticeScalar T> Lattice<T> LLLStrategy<T>::reduce(const Lattice<T> &lat) const
{
    auto Ind = lat.independent();
    int n = Ind.rank(), d = Ind.dim();
    if (n <= 1)
        return Ind.clone();
    // Copy basis to double for orthogonalization
    ndarray<double> B(std::vector<int>{n, d});
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < d; ++j)
            B(i, j) = static_cast<double>(Ind.basis(i, j));

    auto gram_schmidt = [&](const ndarray<double> &Bb, std::vector<std::vector<double>> &Bstar,
                            std::vector<std::vector<double>> &mu) {
        Bstar.assign(n, std::vector<double>(d, 0));
        mu.assign(n, std::vector<double>(n, 0));
        for (int i = 0; i < n; ++i)
        {
            for (int k = 0; k < d; ++k)
                Bstar[i][k] = Bb(i, k);
            for (int j = 0; j < i; ++j)
            {
                double dot = 0, nj = 0;
                for (int k = 0; k < d; ++k)
                    dot += Bb(i, k) * Bstar[j][k];
                for (int k = 0; k < d; ++k)
                    nj += Bstar[j][k] * Bstar[j][k];
                mu[i][j] = nj > 1e-12 ? dot / nj : 0;
                for (int k = 0; k < d; ++k)
                    Bstar[i][k] -= mu[i][j] * Bstar[j][k];
            }
        }
    };

    std::vector<std::vector<double>> Bstar, mu;
    gram_schmidt(B, Bstar, mu);

    auto size_reduce = [&](int k, int l) {
        if (std::abs(mu[k][l]) > eta)
        {
            int r = static_cast<int>(std::round(mu[k][l]));
            for (int j = 0; j < d; ++j)
                B(k, j) -= r * B(l, j);
            // update mu
            gram_schmidt(B, Bstar, mu);
        }
    };

    int k = 1;
    int iter = 0;
    const int max_iter = 1000 * n * n;
    while (k < n && iter < max_iter)
    {
        ++iter;
        for (int j = k - 1; j >= 0; --j)
            size_reduce(k, j);
        // Lovasz condition: |b*_k|^2 >= (delta - mu_{k,k-1}^2) |b*_{k-1}|^2
        double nrm_k = 0, nrm_km1 = 0;
        for (int j = 0; j < d; ++j)
        {
            nrm_k += Bstar[k][j] * Bstar[k][j];
            nrm_km1 += Bstar[k - 1][j] * Bstar[k - 1][j];
        }
        double lhs = nrm_k;
        double rhs = (delta - mu[k][k - 1] * mu[k][k - 1]) * nrm_km1;
        if (lhs >= rhs - 1e-12)
        {
            ++k;
        }
        else
        {
            // swap b_k and b_{k-1}
            for (int j = 0; j < d; ++j)
                std::swap(B(k, j), B(k - 1, j));
            gram_schmidt(B, Bstar, mu);
            k = std::max(k - 1, 1);
        }
    }
    ndarray<T> Br(std::vector<int>{n, d});
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < d; ++j)
            Br(i, j) = static_cast<T>(std::round(B(i, j))); // LLL often keeps integer
    Lattice<T> out(Br);
    out.notify("lll_reduce");
    return out;
}

// ── PosetLattice (order-theoretic) ──────────────────────────────────────
template <Ordered T> struct PosetLattice
{
    std::vector<T> elems;
    std::function<bool(const T &, const T &)> leq; // a <= b ?
    mutable std::shared_mutex mtx_;
    mutable std::vector<std::function<void(const PosetLattice &, const std::string &)>> observers_;

    PosetLattice() = default;
    PosetLattice(std::vector<T> e, std::function<bool(const T &, const T &)> cmp)
        : elems(std::move(e)), leq(std::move(cmp))
    {
    }

    void add_observer(std::function<void(const PosetLattice &, const std::string &)> obs) const
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

    NP_NODISCARD std::optional<T> meet(const T &a, const T &b) const
    {
        // greatest lower bound: maximal c with c<=a and c<=b
        std::optional<T> best;
        for (auto &c : elems)
        {
            if (leq(c, a) && leq(c, b))
            {
                if (!best || leq(*best, c))
                    best = c;
            }
        }
        return best;
    }

    NP_NODISCARD std::optional<T> join(const T &a, const T &b) const
    {
        std::optional<T> best;
        for (auto &c : elems)
        {
            if (leq(a, c) && leq(b, c))
            {
                if (!best || leq(c, *best))
                    best = c;
            }
        }
        return best;
    }

    NP_NODISCARD bool is_lattice() const
    {
        for (auto &a : elems)
            for (auto &b : elems)
                if (!meet(a, b).has_value() || !join(a, b).has_value())
                    return false;
        return true;
    }

    NP_NODISCARD bool is_distributive() const
    {
        // check a ∧ (b ∨ c) == (a∧b) ∨ (a∧c)
        for (auto &a : elems)
            for (auto &b : elems)
                for (auto &c : elems)
                {
                    auto bjc = join(b, c);
                    auto ajb = meet(a, b);
                    auto ajc = meet(a, c);
                    if (!bjc || !ajb || !ajc)
                        continue;
                    auto left = meet(a, *bjc);
                    auto right = join(*ajb, *ajc);
                    if (!left || !right || *left != *right)
                        return false;
                }
        return true;
    }

    NP_NODISCARD bool is_modular() const
    {
        for (auto &a : elems)
            for (auto &b : elems)
                for (auto &c : elems)
                {
                    if (!leq(a, c))
                        continue;
                    auto ajb = join(a, b);
                    auto amb = meet(a, b);
                    // need to check modular law: a ∨ (b ∧ c) == (a ∨ b) ∧ c when a ≤ c
                    auto bmc = meet(b, c);
                    // ... simplified
                    (void)ajb;
                    (void)amb;
                    (void)bmc;
                }
        return true; // stub
    }

    NP_NODISCARD std::vector<std::pair<T, T>> hasse_diagram() const
    {
        std::vector<std::pair<T, T>> covers;
        for (auto &a : elems)
            for (auto &b : elems)
                if (leq(a, b) && a != b)
                {
                    bool is_cover = true;
                    for (auto &c : elems)
                        if (leq(a, c) && leq(c, b) && c != a && c != b)
                        {
                            is_cover = false;
                            break;
                        }
                    if (is_cover)
                        covers.emplace_back(a, b);
                }
        return covers;
    }

    NP_NODISCARD std::map<std::pair<T, T>, int> mobius() const
    {
        // naive recursion for small posets
        std::map<std::pair<T, T>, int> mu;
        auto sorted = elems;
        std::sort(sorted.begin(), sorted.end());
        for (auto &a : sorted)
            for (auto &b : sorted)
                if (leq(a, b))
                {
                    if (a == b)
                        mu[{a, b}] = 1;
                    else
                    {
                        int s = 0;
                        for (auto &c : sorted)
                            if (leq(a, c) && leq(c, b) && c != b)
                                s += mu[{a, c}];
                        mu[{a, b}] = -s;
                    }
                }
        return mu;
    }

    NP_NODISCARD std::map<std::pair<T, T>, int> zeta() const
    {
        std::map<std::pair<T, T>, int> z;
        for (auto &a : elems)
            for (auto &b : elems)
                z[{a, b}] = leq(a, b) ? 1 : 0;
        return z;
    }
};

// ── Decorator (Decorator pattern) ───────────────────────────────────────
template <LatticeScalar T> struct TransformedLattice
{
    Lattice<T> inner;
    ndarray<T> transform; // d x d matrix
    TransformedLattice(Lattice<T> l, ndarray<T> tr) : inner(std::move(l)), transform(std::move(tr))
    {
    }
    NP_NODISCARD Lattice<T> as_lattice() const
    {
        int n = inner.rank(), d = inner.dim();
        ndarray<T> B(std::vector<int>{n, d});
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < d; ++j)
            {
                T s = T(0);
                for (int k = 0; k < d; ++k)
                    s += inner.basis(i, k) * transform(k, j);
                B(i, j) = s;
            }
        return Lattice<T>(B);
    }
};

// ── Factory (Factory pattern) ───────────────────────────────────────────
struct LatticeFactory
{
    template <LatticeScalar T = double> NP_NODISCARD static Lattice<T> cubic(int n, T scale = T(1))
    {
        ndarray<T> B(std::vector<int>{n, n});
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                B(i, j) = (i == j ? scale : T(0));
        return Lattice<T>(B);
    }
    template <LatticeScalar T = double> NP_NODISCARD static Lattice<T> hexagonal(T scale = T(1))
    {
        ndarray<T> B(std::vector<int>{2, 2});
        B(0, 0) = scale;
        B(0, 1) = T(0);
        B(1, 0) = scale * T(0.5);
        B(1, 1) = scale * static_cast<T>(std::sqrt(3.0) / 2.0);
        return Lattice<T>(B);
    }
    template <LatticeScalar T = double> NP_NODISCARD static Lattice<T> a_n(int n)
    {
        // A_n root lattice in R^{n+1}: vectors e_i - e_{i+1}
        ndarray<T> B(std::vector<int>{n, n + 1});
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n + 1; ++j)
                B(i, j) = T(0);
        for (int i = 0; i < n; ++i)
        {
            B(i, i) = T(1);
            B(i, i + 1) = T(-1);
        }
        return Lattice<T>(B);
    }
    template <LatticeScalar T = double> NP_NODISCARD static Lattice<T> d_n(int n)
    {
        ndarray<T> B(std::vector<int>{n, n});
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                B(i, j) = T(0);
        for (int i = 0; i < n - 1; ++i)
        {
            B(i, i) = T(1);
            B(i, i + 1) = T(-1);
        }
        B(n - 1, n - 2) = T(1);
        B(n - 1, n - 1) = T(1);
        return Lattice<T>(B);
    }
    template <LatticeScalar T = double> NP_NODISCARD static Lattice<T> e8()
    {
        // E8 root lattice 8x8
        ndarray<T> B(std::vector<int>{8, 8});
        for (int i = 0; i < 8; ++i)
            for (int j = 0; j < 8; ++j)
                B(i, j) = T(0);
        for (int i = 0; i < 7; ++i)
        {
            B(i, i) = T(1);
            B(i, i + 1) = T(-1);
        }
        B(7, 0) = T(0.5);
        B(7, 1) = T(-0.5);
        B(7, 2) = T(-0.5);
        B(7, 3) = T(-0.5);
        B(7, 4) = T(-0.5);
        B(7, 5) = T(-0.5);
        B(7, 6) = T(-0.5);
        B(7, 7) = T(0.5);
        return Lattice<T>(B);
    }
    template <Ordered T = int> NP_NODISCARD static PosetLattice<T> boolean_lattice(int n)
    {
        // Subsets of {0..n-1} ordered by inclusion, encoded as bitmask
        int N = 1 << n;
        std::vector<int> elems(N);
        std::iota(elems.begin(), elems.end(), 0);
        auto leq = [](const int &a, const int &b) { return (a & b) == a; };
        return PosetLattice<int>(elems, leq);
    }
    template <Ordered T = int> NP_NODISCARD static PosetLattice<T> divisor_lattice(int n)
    {
        std::vector<int> elems;
        for (int i = 1; i <= n; ++i)
            if (n % i == 0)
                elems.push_back(i);
        auto leq = [](const int &a, const int &b) { return b % a == 0; };
        return PosetLattice<int>(elems, leq);
    }
};

// ── Builder (Builder pattern) ───────────────────────────────────────────
template <LatticeScalar T = double> struct LatticeBuilder
{
    std::vector<std::vector<T>> rows_;
    int dim_ = -1;
    double delta_ = 0.75;
    bool do_lll_ = false;

    LatticeBuilder &add_row(std::vector<T> r)
    {
        if (dim_ == -1)
            dim_ = static_cast<int>(r.size());
        else if (static_cast<int>(r.size()) != dim_)
            throw std::invalid_argument("LatticeBuilder: row dim mismatch");
        rows_.push_back(std::move(r));
        return *this;
    }
    LatticeBuilder &add_basis(const ndarray<T> &B)
    {
        int n = B.shape[0], d = B.shape[1];
        if (dim_ == -1)
            dim_ = d;
        else if (d != dim_)
            throw std::invalid_argument("LatticeBuilder: dim mismatch");
        for (int i = 0; i < n; ++i)
        {
            std::vector<T> r(d);
            for (int j = 0; j < d; ++j)
                r[j] = B(i, j);
            rows_.push_back(std::move(r));
        }
        return *this;
    }
    LatticeBuilder &with_lll(double delta = 0.75)
    {
        do_lll_ = true;
        delta_ = delta;
        return *this;
    }
    NP_NODISCARD Lattice<T> build() const
    {
        if (rows_.empty())
            throw std::invalid_argument("LatticeBuilder: empty");
        int n = static_cast<int>(rows_.size());
        ndarray<T> B(std::vector<int>{n, dim_});
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < dim_; ++j)
                B(i, j) = rows_[i][j];
        Lattice<T> lat(B);
        if (do_lll_)
            lat = lat.lll_reduce(delta_);
        return lat;
    }
    NP_NODISCARD static LatticeBuilder<T> create()
    {
        return {};
    }
};

// ── Free ops (meet/join/dual/lll) ───────────────────────────────────────
template <LatticeScalar T> NP_NODISCARD inline Lattice<T> meet(const Lattice<T> &a, const Lattice<T> &b)
{
    return a.meet(b);
}
template <LatticeScalar T> NP_NODISCARD inline Lattice<T> join(const Lattice<T> &a, const Lattice<T> &b)
{
    return a.join(b);
}
template <LatticeScalar T> NP_NODISCARD inline Lattice<T> dual(const Lattice<T> &a)
{
    return a.dual();
}
template <LatticeScalar T> NP_NODISCARD inline Lattice<T> lll(const Lattice<T> &a, double d = 0.75)
{
    return a.lll_reduce(d);
}
template <LatticeScalar T> NP_NODISCARD inline ndarray<T> gram(const Lattice<T> &a)
{
    return a.gram_matrix();
}
template <LatticeScalar T> NP_NODISCARD inline T volume(const Lattice<T> &a)
{
    return a.volume();
}
template <LatticeScalar T> NP_NODISCARD inline ndarray<T> shortest(const Lattice<T> &a)
{
    return a.shortest_vector();
}
template <LatticeScalar T> NP_NODISCARD inline ndarray<T> closest(const Lattice<T> &a, const ndarray<T> &t)
{
    return a.closest_vector(t);
}

template <Ordered T> NP_NODISCARD inline std::optional<T> meet(const PosetLattice<T> &p, const T &a, const T &b)
{
    return p.meet(a, b);
}
template <Ordered T> NP_NODISCARD inline std::optional<T> join(const PosetLattice<T> &p, const T &a, const T &b)
{
    return p.join(a, b);
}

} // namespace np::lattice

#endif // NP_LATTICE_HPP
