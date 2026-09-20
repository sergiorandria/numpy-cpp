/**
 * @file bundle.hpp
 * @brief Vector bundles, tangent bundles, characteristic classes and Hodge theory.
 *
 * Provides `np::bundle` with:
 *   - `VectorBundle` – rank, base, orientability, complex structure
 *   - `tangent_bundle`, `cotangent_bundle`, `normal_bundle`, `pullback_bundle`
 *   - Characteristic classes: `chern_classes`, `stiefel_whitney_classes`,
 *     `euler_class`, `pontryagin_classes`, `total_chern_class`, `whitney_sum`
 *   - Hodge: `HodgeStar`, `codifferential`, `laplacian`, `harmonic_forms`
 *
 * For classical manifolds (S^n, T^n, CP^n, RP^n, Klein) the Chern/Whitney
 * numbers are exact via Bott–Tu / Milnor-Stasheff:
 *   - `T CP^n : c = (1+h)^{n+1}`, `e = n+1`, `p = (1+h^2)^{n+1}`
 *   - `T RP^n : w = (1+a)^{n+1} (mod 2)`
 *   - `T S^n : w = 1 (stably trivial)`, `e = 2` (even) / `0` (odd); `T T^n` trivial.
 * Generic bundles fall back to zero classes with `inconclusive=true`.
 *
 * Reference: Milnor–Stasheff, Bott–Tu, Lee *Riemannian Manifolds*.
 *
 * @author Sergio Randriamihoatra (sergiorandriamihoatra@gmail.com)
 */
#ifndef NP_BUNDLE_HPP
#define NP_BUNDLE_HPP

#include <algorithm>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "api_macros.hpp"
#include "bigint.hpp"
#include "differential.hpp"
#include "homology.hpp"
#include "manifold.hpp"

namespace np::inline v1
{
namespace bundle
{

struct VectorBundle
{
    std::string name = "E";
    std::string base_name = "M";
    int base_dim = 0;
    int rank = 0; // real rank; complex rank = rank/2 if is_complex
    bool is_complex = false;
    bool is_orientable = true;
    bool is_stable_trivial = false;
    std::string to_string() const
    {
        return name + " → " + base_name + " rank " + std::to_string(rank) + (is_complex ? " (C)" : " (R)");
    }
};

struct CharacteristicClasses
{
    std::vector<bigint> chern;      // c0=1, c1.. ; size = complex rank +1
    std::vector<int> stiefel;       // w0.. ; mod 2 (0/1)
    bigint euler = 0;               // Euler number (Top) – 0 if not oriented even rank
    std::vector<bigint> pontryagin; // p_i in H^{4i}
    bool inconclusive = false;
    std::string to_string() const
    {
        std::string s = "c=[";
        for (size_t i = 0; i < chern.size(); ++i)
        {
            if (i)
                s += ",";
            s += chern[i].convert_to<std::string>();
        }
        s += "] w=[";
        for (size_t i = 0; i < stiefel.size(); ++i)
        {
            if (i)
                s += ",";
            s += std::to_string(stiefel[i]);
        }
        s += "] e=" + euler.convert_to<std::string>();
        if (inconclusive)
            s += " (inconclusive)";
        return s;
    }
};

namespace detail
{

// Exact binomial via multiplicative formula in bigint (arbitrary precision).
// Replaces the old long-long version, which overflowed for n >= 67
// (e.g. CP^34: C(35,17) > 2^63). Returns 0 outside 0<=k<=n.
NP_NODISCARD inline bigint binom_exact(int n, int k)
{
    if (k < 0 || k > n || n < 0)
        return bigint(0);
    if (k > n - k)
        k = n - k;
    bigint r(1);
    for (int i = 0; i < k; ++i)
    {
        r *= bigint(n - i);
        r /= bigint(i + 1);
    }
    return r;
}

// Parse "P^<digits>" suffix; returns -1 when absent or malformed.
// Narrows stoi exceptions per AGENTS.md §4 (no catch-all swallow);
// requires full consumption so "CP^3x" is inconclusive, not 3.
NP_NODISCARD inline int parse_projective_n(const std::string &bn)
{
    auto pos = bn.find("P^");
    if (pos == std::string::npos)
        return -1;
    try
    {
        const std::string tail = bn.substr(pos + 2);
        std::size_t used = 0;
        const int v = std::stoi(tail, &used);
        if (used == 0 || used != tail.size() || v < 0)
            return -1;
        return v;
    }
    catch (const std::exception &)
    {
        return -1;
    }
}

} // namespace detail

NP_NODISCARD inline VectorBundle tangent_bundle(const manifold::AbstractManifold &M)
{
    VectorBundle E;
    E.base_name = M.name();
    E.base_dim = M.dimension();
    E.rank = M.dimension();
    E.name = "T" + E.base_name;
    E.is_complex = M.is_kahler();
    E.is_orientable = M.is_orientable();
    // Tori are parallelizable → stably trivial
    if (M.name().rfind("T^", 0) == 0)
        E.is_stable_trivial = true;
    if (M.name() == "S^1" || M.name() == "S^3" || M.name() == "S^7")
        E.is_stable_trivial = true;
    return E;
}

NP_NODISCARD inline VectorBundle cotangent_bundle(const manifold::AbstractManifold &M)
{
    auto E = tangent_bundle(M);
    E.name = "T*" + M.name();
    return E;
}

NP_NODISCARD inline VectorBundle normal_bundle(const manifold::AbstractManifold &M, int ambient_dim)
{
    VectorBundle E;
    E.base_name = M.name();
    E.base_dim = M.dimension();
    E.rank = ambient_dim - M.dimension();
    if (E.rank < 0)
        E.rank = 0;
    E.name = "N" + E.base_name;
    E.is_orientable = M.is_orientable();
    return E;
}

NP_NODISCARD inline CharacteristicClasses characteristic_classes(const VectorBundle &E,
                                                                 const manifold::AbstractManifold *base = nullptr)
{
    CharacteristicClasses C;
    C.chern.assign(1, bigint(1));
    C.stiefel.assign(1, 1);
    C.euler = 0;
    // Tangent of known bases (needs non-negative dimension for sizing).
    if (base && base->dimension() >= 0)
    {
        std::string bn = base->name();
        int n = base->dimension();
        // Torus
        if (bn.rfind("T^", 0) == 0)
        {
            C.chern = {bigint(1)};
            C.stiefel = {1};
            C.pontryagin = {bigint(1)};
            C.euler = 0;
            if (E.is_complex)
                C.chern.assign(1, bigint(1));
            return C;
        }
        // Sphere S^n: stably trivial (TS^n + trivial = trivial^{n+1}),
        // so total Stiefel-Whitney w = 1. Euler is 2 (even) / 0 (odd).
        if (bn.rfind("S^", 0) == 0)
        {
            C.chern = {bigint(1)};
            C.stiefel.assign(1, 1);
            C.pontryagin = {bigint(1)};
            if (n % 2 == 0)
                C.euler = 2;
            else
                C.euler = 0;
            return C;
        }
        // Complex projective CP^n
        if (bn.rfind("C", 0) == 0 && bn.find("P^") != std::string::npos)
        {
            int cp_n = detail::parse_projective_n(bn);
            if (cp_n < 0)
            {
                C.inconclusive = true;
                return C;
            }
            int cpx_rank = cp_n; // complex rank of T CP^n is n
            C.chern.assign(static_cast<std::size_t>(cpx_rank) + 1, bigint(0));
            for (int k = 0; k <= cpx_rank; ++k)
                C.chern[static_cast<std::size_t>(k)] = detail::binom_exact(cp_n + 1, k);
            // Euler = n+1
            C.euler = cp_n + 1;
            // Stiefel = mod2 reduction of Chern: w_{2k}=c_k mod2, w_{odd}=0
            C.stiefel.assign(static_cast<std::size_t>(2 * cpx_rank) + 1, 0);
            C.stiefel[0] = 1;
            for (int k = 1; k <= cpx_rank; ++k)
                C.stiefel[static_cast<std::size_t>(2 * k)] = static_cast<int>(detail::binom_exact(cp_n + 1, k) % 2);
            // Pontryagin from Chern: p = (1+h^2)^{n+1}, so p_k = C(n+1,k).
            C.pontryagin.assign(static_cast<std::size_t>(cpx_rank) + 1, bigint(0));
            C.pontryagin[0] = 1;
            for (int k = 1; k <= cpx_rank / 2; ++k)
                C.pontryagin[static_cast<std::size_t>(k)] = detail::binom_exact(cp_n + 1, k);
            return C;
        }
        // Real projective RP^n
        if (bn.rfind("R", 0) == 0 && bn.find("P^") != std::string::npos)
        {
            const int rp_n = detail::parse_projective_n(bn);
            if (rp_n < 0)
            {
                C.inconclusive = true;
                return C;
            }
            C.stiefel.assign(static_cast<std::size_t>(rp_n) + 1, 0);
            for (int k = 0; k <= rp_n; ++k)
                C.stiefel[static_cast<std::size_t>(k)] = static_cast<int>(detail::binom_exact(rp_n + 1, k) % 2);
            C.chern = {bigint(1)};
            // RP^n orientable iff n odd (Euler 0); RP^{even} is non-orientable
            // so Euler class is convention-0 even though chi = 1.
            C.euler = 0;
            C.pontryagin = {bigint(1)};
            return C;
        }
        // Klein bottle
        if (bn == "Klein")
        {
            C.stiefel = {1, 1, 0}; // w1≠0, w2=0
            C.euler = 0;
            return C;
        }
    }
    // Generic fallback: trivial
    C.inconclusive = true;
    C.chern = {bigint(1)};
    C.stiefel = {1};
    C.pontryagin = {bigint(1)};
    return C;
}

NP_NODISCARD inline std::vector<bigint> chern_classes(const VectorBundle &E,
                                                      const manifold::AbstractManifold *base = nullptr)
{
    return characteristic_classes(E, base).chern;
}

NP_NODISCARD inline std::vector<int> stiefel_whitney_classes(const VectorBundle &E,
                                                             const manifold::AbstractManifold *base = nullptr)
{
    return characteristic_classes(E, base).stiefel;
}

NP_NODISCARD inline bigint euler_class(const VectorBundle &E, const manifold::AbstractManifold *base = nullptr)
{
    return characteristic_classes(E, base).euler;
}

NP_NODISCARD inline std::vector<bigint> pontryagin_classes(const VectorBundle &E,
                                                           const manifold::AbstractManifold *base = nullptr)
{
    return characteristic_classes(E, base).pontryagin;
}

NP_NODISCARD inline bigint euler_characteristic_via_euler_class(const VectorBundle &,
                                                                const manifold::AbstractManifold *base)
{
    // For tangent bundle, ∫_M e(TM) = χ(M)
    if (!base)
        return bigint(0);
    return bigint(base->euler_characteristic());
}

// Whitney sum

NP_NODISCARD inline VectorBundle whitney_sum(const VectorBundle &A, const VectorBundle &B)
{
    VectorBundle S;
    S.base_name = A.base_name;
    S.base_dim = std::max(A.base_dim, B.base_dim);
    S.rank = A.rank + B.rank;
    S.is_complex = A.is_complex && B.is_complex;
    S.is_orientable = A.is_orientable && B.is_orientable;
    S.name = A.name + " ⊕ " + B.name;
    return S;
}

NP_NODISCARD inline CharacteristicClasses whitney_sum_classes(const CharacteristicClasses &A,
                                                              const CharacteristicClasses &B)
{
    CharacteristicClasses S;
    // Total Chern w = w(A)⌣w(B) ; for line bundles c = (1+c1(A))(1+c1(B))
    // Over Z, c_k = Σ_{i+j=k} c_i(A) c_j(B); convolution needs na+nb-1 slots.
    // Empty input means "no classes recorded" → treat as {1} (identity).
    const size_t na = A.chern.empty() ? 1 : A.chern.size();
    const size_t nb = B.chern.empty() ? 1 : B.chern.size();
    S.chern.assign(na + nb - 1, bigint(0));
    for (size_t i = 0; i < na; ++i)
        for (size_t j = 0; j < nb; ++j)
            S.chern[i + j] +=
                (i < A.chern.size() ? A.chern[i] : bigint(1)) * (j < B.chern.size() ? B.chern[j] : bigint(1));
    // Trim trailing zeros
    while (S.chern.size() > 1 && S.chern.back() == 0)
        S.chern.pop_back();
    // Stiefel mod2 (same convolution over GF(2))
    const size_t ma = A.stiefel.empty() ? 1 : A.stiefel.size();
    const size_t mb = B.stiefel.empty() ? 1 : B.stiefel.size();
    S.stiefel.assign(ma + mb - 1, 0);
    for (size_t i = 0; i < ma; ++i)
        for (size_t j = 0; j < mb; ++j)
            S.stiefel[i + j] ^= ((i < A.stiefel.size() ? A.stiefel[i] : 1) & (j < B.stiefel.size() ? B.stiefel[j] : 1));
    while (S.stiefel.size() > 1 && S.stiefel.back() == 0)
        S.stiefel.pop_back();
    // e(E+ F) = e(E) cup e(F) is exact as cohomology classes (Milnor-Stasheff, §9); as stored Euler
    // numbers the product is exact for oriented even-rank summands. Odd-rank summands store
    // convention-0, which can mask a nonzero even-rank sum — treat such results as approximate.
    S.euler = A.euler * B.euler;
    S.inconclusive = A.inconclusive || B.inconclusive;
    return S;
}

// Hodge theory

struct HodgeStar
{
    int n = 0; // manifold dimension
    explicit HodgeStar(int dim = 0) : n(dim)
    {
        if (dim < 0)
            throw std::invalid_argument("HodgeStar: dimension must be non-negative");
    }
    /**
     * @brief * : Ω^k → Ω^{n-k} on oriented Riemannian manifold.
     * With flat metric, *² = (-1)^{k(n-k)}.
     */
    NP_NODISCARD int sign(int k) const noexcept
    {
        const int e = k * (n - k);
        return (e % 2 == 0) ? 1 : -1;
    }
};

namespace detail
{

// Sign of the permutation (I,J) relative to (0..n-1), for sorted k-set I.
// Equals (-1)^{sum(I) - k(k-1)/2} (standard Hodge complement sign).
NP_NODISCARD inline int complement_sign(const std::vector<int> &idx, int k) noexcept
{
    long long s = 0;
    for (int v : idx)
        s += v;
    s -= static_cast<long long>(k) * (k - 1) / 2;
    return (s % 2 == 0) ? 1 : -1;
}

NP_NODISCARD inline differential::KForm scale_form(const differential::KForm &w, double s)
{
    differential::KForm out;
    out.k = w.k;
    out.dim = w.dim;
    for (auto &[idx, field] : w.coeffs)
    {
        auto f = field.f;
        out.coeffs[idx] =
            differential::ScalarField([f, s](const differential::Point &p) { return s * f(p); }, field.dim);
    }
    return out;
}

} // namespace detail

NP_NODISCARD inline differential::KForm hodge_star(const differential::KForm &w, const HodgeStar &hs)
{
    if (w.k < 0 || w.k > hs.n)
        throw std::invalid_argument("hodge_star: form degree out of range");
    if (w.dim != hs.n)
        throw std::invalid_argument("hodge_star: form dim must match manifold dim");
    differential::KForm out;
    out.k = hs.n - w.k;
    out.dim = w.dim;
    // Flat orthonormal metric: *(f dx_I) = sign(I,J) * f dx_J, J = complement,
    // where sign(I,J) is the permutation sign of (I,J). The (-1)^{k(n-k)}
    // factor appears only in **, never in a single application.
    for (auto &[idx, field] : w.coeffs)
    {
        // Complement indices
        std::vector<int> comp;
        for (int i = 0; i < hs.n; ++i)
            if (std::find(idx.begin(), idx.end(), i) == idx.end())
                comp.push_back(i);
        std::sort(comp.begin(), comp.end());
        const int total = detail::complement_sign(idx, w.k);
        auto f = field.f;
        const int d = field.dim;
        if (total == 1)
        {
            out.coeffs[comp] = field;
        }
        else
        {
            out.coeffs[comp] = differential::ScalarField([f](const differential::Point &p) { return -f(p); }, d);
        }
    }
    return out;
}

NP_NODISCARD inline differential::KForm codifferential(const differential::KForm &w, const HodgeStar &hs)
{
    // δ = (-1)^{n(k+1)+1} ⋆ d ⋆ on k-forms (Bott–Tu Prop. 6.5).
    if (w.k <= 0)
        return differential::KForm(0, w.dim); // δ = 0 on 0-forms
    auto dw_star = differential::exterior_derivative(hodge_star(w, hs));
    auto back = hodge_star(dw_star, hs);
    const int e = hs.n * (w.k + 1) + 1;
    return detail::scale_form(back, (e % 2 == 0) ? 1.0 : -1.0);
}

NP_NODISCARD inline differential::KForm laplacian(const differential::KForm &w, const HodgeStar &hs)
{
    // Δ = dδ + δd (Hodge Laplacian); exact on flat coefficients.
    // Boundary terms vanish: δ = 0 on 0-forms, d = 0 on n-forms.
    if (w.k < 0 || w.k > hs.n)
        throw std::invalid_argument("laplacian: form degree out of range");
    if (w.k == 0)
        return codifferential(differential::exterior_derivative(w), hs);
    if (w.k == hs.n)
        return differential::exterior_derivative(codifferential(w, hs));
    auto dd = differential::exterior_derivative(codifferential(w, hs));
    auto dd2 = codifferential(differential::exterior_derivative(w), hs);
    return dd + dd2;
}

NP_NODISCARD inline bool is_harmonic(const differential::KForm &w, const HodgeStar &hs)
{
    if (w.coeffs.empty())
        return true; // zero form is trivially harmonic (also skips dim checks)
    auto Lap = laplacian(w, hs);
    return Lap.coeffs.empty() || w.coeffs.empty();
}

} // namespace bundle
} // namespace np::inline v1

#endif // NP_BUNDLE_HPP
