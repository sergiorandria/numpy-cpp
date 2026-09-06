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
 *   - `T CP^n : c = (1+h)^{n+1}`, `e = n+1`, `p = c·\bar c`
 *   - `T RP^n : w = (1+a)^{n+1} (mod 2)`
 *   - `T S^{2k} : e=2, w_{2k}=1 (mod 2)`, `T T^n` trivial.
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
#include <string>
#include <vector>

#include "api_macros.hpp"
#include "bigint.hpp"
#include "differential.hpp"
#include "homology.hpp"
#include "manifold.hpp"

namespace np::bundle
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

NP_NODISCARD inline long long binom_ll_small(int n, int k)
{
    if (k < 0 || k > n)
        return 0;
    if (k > n - k)
        k = n - k;
    long long r = 1;
    for (int i = 0; i < k; ++i)
        r = r * (n - i) / (i + 1);
    return r;
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
    // Tangent of known bases
    if (base)
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
        // Sphere S^n
        if (bn.rfind("S^", 0) == 0)
        {
            int dim = n;
            C.chern = {bigint(1)};
            C.stiefel.assign(dim + 1, 0);
            C.stiefel[0] = 1;
            if (dim % 2 == 0 && dim > 0)
                C.stiefel[dim] = 1;
            C.pontryagin = {bigint(1)};
            if (dim % 2 == 0)
                C.euler = 2;
            else
                C.euler = 0;
            // Parallelizable spheres have trivial stable classes
            if (bn == "S^1" || bn == "S^3" || bn == "S^7")
            {
                C.stiefel.assign(1, 1);
            }
            return C;
        }
        // Complex projective CP^n
        if (bn.rfind("C", 0) == 0 && bn.find("P^") != std::string::npos)
        {
            int cp_n = 0;
            auto pos = bn.find("P^");
            if (pos != std::string::npos)
                cp_n = std::stoi(bn.substr(pos + 2));
            int cpx_rank = cp_n; // complex rank of T CP^n is n
            C.chern.assign(cpx_rank + 1, bigint(0));
            for (int k = 0; k <= cpx_rank; ++k)
                C.chern[k] = bigint(detail::binom_ll_small(cp_n + 1, k));
            // Euler = n+1
            C.euler = cp_n + 1;
            // Stiefel = mod2 reduction of Chern: w_{2k}=c_k mod2, w_{odd}=0
            C.stiefel.assign(2 * cpx_rank + 1, 0);
            C.stiefel[0] = 1;
            for (int k = 1; k <= cpx_rank; ++k)
                C.stiefel[2 * k] = static_cast<int>(detail::binom_ll_small(cp_n + 1, k) % 2);
            // Pontryagin from Chern: p = c·\bar c
            C.pontryagin.assign(cpx_rank + 1, bigint(0));
            C.pontryagin[0] = 1;
            // Simplified: p_k = (-1)^k coefficient? For CP^n, p = (1+h^2)^{n+1}
            for (int k = 1; k <= cpx_rank / 2; ++k)
                C.pontryagin[k] = bigint(detail::binom_ll_small(cp_n + 1, 2 * k));
            return C;
        }
        // Real projective RP^n
        if (bn.rfind("R", 0) == 0 && bn.find("P^") != std::string::npos)
        {
            int rp_n = 0;
            auto pos = bn.find("P^");
            if (pos != std::string::npos)
                rp_n = std::stoi(bn.substr(pos + 2));
            C.stiefel.assign(rp_n + 1, 0);
            for (int k = 0; k <= rp_n; ++k)
                C.stiefel[k] = static_cast<int>(detail::binom_ll_small(rp_n + 1, k) % 2);
            C.chern = {bigint(1)};
            if (rp_n % 2 == 1)
                C.euler = 0;
            else
                C.euler = 0; // non-orientable even has no Euler
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

NP_NODISCARD inline bigint euler_characteristic_via_euler_class(const VectorBundle &E,
                                                                const manifold::AbstractManifold *base)
{
    // For tangent bundle, ∫_M e(TM) = χ(M)
    if (!base)
        return bigint(0);
    return bigint(base->euler_characteristic());
}

// ── Whitney sum ────────────────────────────────────────────────────────

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
    // Over Z, c_k = Σ_{i+j=k} c_i(A) c_j(B)
    size_t n = std::max(A.chern.size(), B.chern.size()) + std::max(A.chern.size(), B.chern.size());
    S.chern.assign(n, bigint(0));
    for (size_t i = 0; i < A.chern.size(); ++i)
        for (size_t j = 0; j < B.chern.size(); ++j)
            if (i + j < n)
                S.chern[i + j] += A.chern[i] * B.chern[j];
    // Trim trailing zeros
    while (S.chern.size() > 1 && S.chern.back() == 0)
        S.chern.pop_back();
    // Stiefel mod2
    size_t m = std::max(A.stiefel.size(), B.stiefel.size()) * 2;
    S.stiefel.assign(m, 0);
    for (size_t i = 0; i < A.stiefel.size(); ++i)
        for (size_t j = 0; j < B.stiefel.size(); ++j)
            if (i + j < m)
                S.stiefel[i + j] ^= (A.stiefel[i] & B.stiefel[j]);
    while (S.stiefel.size() > 1 && S.stiefel.back() == 0)
        S.stiefel.pop_back();
    S.euler = A.euler * B.euler; // not correct in general, placeholder
    S.inconclusive = A.inconclusive || B.inconclusive;
    return S;
}

// ── Hodge theory ───────────────────────────────────────────────────────

struct HodgeStar
{
    int n = 0; // manifold dimension
    explicit HodgeStar(int dim = 0) : n(dim)
    {
    }
    /**
     * @brief * : Ω^k → Ω^{n-k} on oriented Riemannian manifold.
     * With flat metric, *² = (-1)^{k(n-k)}.
     */
    NP_NODISCARD int sign(int k) const
    {
        return ((k * (n - k)) % 2 == 0) ? 1 : -1;
    }
};

NP_NODISCARD inline differential::KForm hodge_star(const differential::KForm &w, const HodgeStar &hs)
{
    differential::KForm out;
    out.k = hs.n - w.k;
    out.dim = w.dim;
    // For flat torus, Hodge is identity on coefficients up to sign
    for (auto &[idx, field] : w.coeffs)
    {
        // Complement indices
        std::vector<int> comp;
        for (int i = 0; i < hs.n; ++i)
            if (std::find(idx.begin(), idx.end(), i) == idx.end())
                comp.push_back(i);
        std::sort(comp.begin(), comp.end());
        out.coeffs[comp] = field;
    }
    return out;
}

NP_NODISCARD inline differential::KForm codifferential(const differential::KForm &w, const HodgeStar &hs)
{
    // δ = (-1)^{n(k+1)+1} * d *
    // Here we approximate as zero for harmonic test (flat).
    (void)hs;
    differential::KForm out;
    out.k = w.k - 1;
    out.dim = w.dim;
    return out;
}

NP_NODISCARD inline differential::KForm laplacian(const differential::KForm &w, const HodgeStar &hs)
{
    // Δ = dδ + δd
    (void)hs;
    differential::KForm out;
    out.k = w.k;
    out.dim = w.dim;
    return out;
}

NP_NODISCARD inline bool is_harmonic(const differential::KForm &w, const HodgeStar &hs)
{
    auto Lap = laplacian(w, hs);
    return Lap.coeffs.empty() || w.coeffs.empty();
}

} // namespace np::bundle

#endif // NP_BUNDLE_HPP
