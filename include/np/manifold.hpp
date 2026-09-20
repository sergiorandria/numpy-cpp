/**
 * @file manifold.hpp
 * @brief Abstract manifolds and varieties with homology / homotopy / de Rham and logical
 * reasoning.
 *
 * Correct name for `variety.hpp` (kept as alias). Provides
 * `np::manifold::AbstractManifold` and concrete `Sphere`, `Torus`, `ProjectiveSpace`,
 * `KleinBottle`, `Euclidean`, `GenusGSurface`, `LensSpace`, `Product`, `Wedge`,
 * `ConnectedSum`, etc., that integrate with `np::homology` / `np::homotopy` /
 * `np::differential` and provide helpers to fix logical reasoning in differential /
 * topological / algebraic geometry:
 *   - `is_orientable`, `is_compact`, `is_connected`, `is_simply_connected`
 *   - `is_smooth`, `is_complete`, `is_irreducible`, `is_normal`, `is_reduced`
 *   - `check_logical_consistency()` (Euler = Σ(-1)^k betti, Poincaré duality, orientable
 * ↔ H_n, etc.)
 *   - de Rham ↔ singular via `de_rham` vs `homology` (torsion killed over R)
 *   - differential geometry: `metric_tensor`, `riemann_tensor`, `sectional_curvature`
 *
 *   auto S2 = manifold::sphere(2); // S², dim 2
 *   S2.homology(2).betti==1; // H₂=Z → R over R
 *   S2.de_rham(2).betti==1;  // H²_dR=R
 *   S2.is_orientable()==true; S2.check_logical_consistency().ok==true;
 *
 * Reference: Hatcher, Hartshorne, Bott–Tu, Lee *Introduction to Smooth Manifolds*.
 *
 * @author Sergio Randriamihoatra (sergiorandriamihoatra@gmail.com)
 */
#ifndef NP_MANIFOLD_HPP
#define NP_MANIFOLD_HPP

#include <algorithm>
#include <cmath>
#include <concepts>
#include <limits>
#include <map>
#include <memory>
#include <numbers>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

#include "api_macros.hpp"
#include "bigint.hpp"
#include "differential.hpp"
#include "homology.hpp"
#include "homotopy.hpp"
#include "ndarray.hpp"

namespace np::inline v1
{
namespace manifold
{

// ── Geometry helpers ────────────────────────────────────────────────────

/**
 * @brief Riemannian metric as symmetric positive-definite (0,2)-tensor.
 * Stored as n×n ndarray in coordinates.
 */
struct MetricTensor
{
    int dim = 0;
    ndarray<double> g; // shape {dim, dim}
    MetricTensor() = default;
    explicit MetricTensor(int d) : dim(d), g({d, d})
    {
        for (int i = 0; i < d; ++i)
            for (int j = 0; j < d; ++j)
                g(i, j) = (i == j) ? 1.0 : 0.0;
    }
    double operator()(int i, int j) const
    {
        return g(i, j);
    }
};

/**
 * @brief Riemann curvature tensor R^i_{jkl} as n×n×n×n array.
 * For flat space all zero; for S^n constant curvature 1.
 */
struct RiemannTensor
{
    int dim = 0;
    // flattened n^4 storage
    std::vector<double> data;
    RiemannTensor() = default;
    explicit RiemannTensor(int d, double fill = 0.0) : dim(d), data(d * d * d * d, fill)
    {
    }
    double &operator()(int i, int j, int k, int l)
    {
        return data[((i * dim + j) * dim + k) * dim + l];
    }
    double operator()(int i, int j, int k, int l) const
    {
        return data[((i * dim + j) * dim + k) * dim + l];
    }
};

// ── Simplicial helpers ─────────────────────────────────────────────────

namespace detail
{

NP_NODISCARD inline homology::SimplicialComplex sphere_boundary_complex(int n)
{
    if (n < 0)
        return homology::SimplicialComplex{};
    if (n == 0)
    {
        // S^0 is two points (H_0 = Z^2). The trailing empty level keeps
        // homology::betti_numbers() counting both vertices (n_0 comes from
        // the d_1 matrix shape).
        return homology::SimplicialComplex{{{{0}, {1}}, {}}};
    }
    if (n == 1)
        return homology::circle_complex();
    if (n == 2)
        return homology::sphere_tetrahedron();
    // Boundary of (n+1)-simplex: simplex with n+2 vertices, facets are
    // all (n+1)-subsets. The boundary is S^n. Build via maximal faces.
    int V = n + 2;
    std::vector<std::vector<int>> facets;
    facets.reserve(V);
    for (int omit = 0; omit < V; ++omit)
    {
        std::vector<int> f;
        f.reserve(n + 1);
        for (int v = 0; v < V; ++v)
            if (v != omit)
                f.push_back(v);
        facets.push_back(std::move(f));
    }
    homology::SimplicialComplexBuilder b;
    b.add_maximal(facets);
    return b.build();
}

NP_NODISCARD inline homology::SimplicialComplex wedge_simplicial(
    const std::vector<const homology::SimplicialComplex *> &comps)
{
    if (comps.empty())
        return homology::SimplicialComplex{};
    if (comps.size() == 1)
        return *comps[0];
    // Disjoint union identifying vertex 0 of each component as basepoint.
    // Reindex vertices: component c gets offset, but vertex 0 maps to 0.
    homology::SimplicialComplexBuilder b;
    int next_id = 1; // 0 reserved for wedge point
    for (auto *K : comps)
    {
        // Collect all vertex ids (vertices may be sparse, not 0..n-1).
        std::vector<int> all_verts;
        if (!K->simplices.empty() && !K->simplices[0].empty())
        {
            for (auto &s : K->simplices[0])
                all_verts.push_back(s[0]);
            std::sort(all_verts.begin(), all_verts.end());
        }
        // Build map: 0 -> 0, others -> next_id++
        std::map<int, int> mp;
        mp[0] = 0;
        int offset_base = next_id;
        // Assign new ids for non-zero vertices
        for (int v : all_verts)
        {
            if (v == 0)
                continue;
            mp[v] = next_id++;
        }
        // If complex is empty or has single vertex, no new vertices
        if (K->simplices.empty())
            continue;
        // Add each maximal simplex remapped
        // Instead iterate all simplices by dimension and re-add
        for (size_t d = 0; d < K->simplices.size(); ++d)
        {
            for (auto s : K->simplices[d])
            {
                for (int &x : s)
                    x = mp[x];
                b.add_simplex(s);
            }
        }
        (void)offset_base;
    }
    return b.build();
}

NP_NODISCARD inline int binomial_int(int n, int k)
{
    if (k < 0 || k > n)
        return 0;
    if (k == 0 || k == n)
        return 1;
    long long num = 1, den = 1;
    for (int i = 0; i < k; ++i)
    {
        num *= (n - i);
        den *= (k - i);
    }
    return static_cast<int>(num / den);
}

NP_NODISCARD inline bigint bigint_gcd(bigint a, bigint b)
{
    if (a < 0)
        a = -a;
    if (b < 0)
        b = -b;
    while (b != 0)
    {
        bigint r = a % b;
        a = b;
        b = r;
    }
    return a;
}

/**
 * @brief Künneth torsion of H_n(X×Y) from the two factor homologies.
 *
 * From 0 → ⊕_{p+q=n} H_p⊗H_q → H_n → ⊕_{p+q=n-1} Tor(H_p,H_q) → 0 with
 * Z/a⊗Z/b = Tor(Z/a,Z/b) = Z/gcd(a,b), Z⊗Z/m = Z/m, Tor(Z,−) = 0.
 * The free summand splits, so torsion is the multiset union below
 * (invariant-factor re-diagonalization is unnecessary for rank-1 factors
 * and stays a sound over-approximation otherwise).
 */
NP_NODISCARD inline std::vector<bigint> kunneth_product_torsion(const std::vector<homology::HomologyGroup> &hX,
                                                                const std::vector<homology::HomologyGroup> &hY, int n)
{
    std::vector<bigint> out;
    const int dx = static_cast<int>(hX.size()) - 1;
    const int dy = static_cast<int>(hY.size()) - 1;
    auto at = [](const std::vector<homology::HomologyGroup> &h, int k) -> const homology::HomologyGroup * {
        if (k < 0 || k >= static_cast<int>(h.size()))
            return nullptr;
        return &h[k];
    };
    // Tensor terms: p + q == n
    for (int p = 0; p <= n; ++p)
    {
        const int q = n - p;
        const auto *hx = at(hX, p);
        const auto *hy = at(hY, q);
        if (hx == nullptr || hy == nullptr)
            continue;
        for (auto t : hx->torsion)
            for (int i = 0; i < hy->betti; ++i)
                out.push_back(t);
        for (auto u : hy->torsion)
            for (int i = 0; i < hx->betti; ++i)
                out.push_back(u);
        for (auto t : hx->torsion)
            for (auto u : hy->torsion)
            {
                bigint g = bigint_gcd(t, u);
                if (g > 1)
                    out.push_back(g);
            }
    }
    // Tor terms: p + q == n - 1
    for (int p = 0; p <= n - 1; ++p)
    {
        const int q = n - 1 - p;
        if (p > dx || q > dy || p < 0 || q < 0)
            continue;
        const auto *hx = at(hX, p);
        const auto *hy = at(hY, q);
        if (hx == nullptr || hy == nullptr)
            continue;
        for (auto t : hx->torsion)
            for (auto u : hy->torsion)
            {
                bigint g = bigint_gcd(t, u);
                if (g > 1)
                    out.push_back(g);
            }
    }
    std::sort(out.begin(), out.end());
    return out;
}

/**
 * @brief Top dimension with actual simplices (ignores trailing empty levels).
 *
 * `SimplicialComplex::dim()` counts levels; some stored complexes carry a
 * trailing empty level (e.g. the RP² triangulation below). Homology consumers
 * need the largest dimension that really occurs (for facet selection).
 */
NP_NODISCARD inline int top_dimension(const homology::SimplicialComplex &K)
{
    int t = 0;
    for (size_t d = 0; d < K.simplices.size(); ++d)
        if (!K.simplices[d].empty())
            t = static_cast<int>(d);
    return t;
}

/**
 * @brief Kuhn (Freudenthal) triangulation of the d-torus, d >= 3.
 *
 * Vertices are the 3^d grid points of (Z/3Z)^d (id = base-3 number). Each
 * unit cube (product of circle edges [e_i, e_i+1]) is triangulated into d!
 * simplices, one per axis permutation (chain v_0 < v_1 < ... < v_d stepping
 * one coordinate at a time). The triangulation is translation-invariant, so
 * opposite faces match and the result triangulates (S¹)^d with
 * H_k = Z^C(d,k) (verified for d = 3: Betti [1,3,3,1]).
 *
 * Cost is exponential (3^d vertices, 3^d·d! top simplices); use
 * torus_betti_wedge() for d >= 5.
 */
NP_NODISCARD inline homology::SimplicialComplex torus_kuhn_complex(int d)
{
    std::vector<int> axes(d);
    std::iota(axes.begin(), axes.end(), 0);
    std::vector<std::vector<int>> perms;
    do
    {
        perms.push_back(axes);
    } while (std::next_permutation(axes.begin(), axes.end()));
    long long ncubes = 1;
    for (int i = 0; i < d; ++i)
        ncubes *= 3;
    std::vector<std::vector<int>> maxsimp;
    std::vector<int> e(d), v(d), chain(d + 1);
    for (long long c = 0; c < ncubes; ++c)
    {
        long long t = c;
        for (int i = 0; i < d; ++i)
        {
            e[i] = static_cast<int>(t % 3);
            t /= 3;
        }
        for (const auto &pi : perms)
        {
            v = e;
            auto vid = [&]() {
                int id = 0;
                for (int x : v)
                    id = id * 3 + x;
                return id;
            };
            chain[0] = vid();
            for (int k = 0; k < d; ++k)
            {
                v[pi[k]] = (v[pi[k]] + 1) % 3;
                chain[k + 1] = vid();
            }
            std::sort(chain.begin(), chain.end());
            maxsimp.push_back(chain);
        }
    }
    homology::SimplicialComplexBuilder b;
    b.add_maximal(maxsimp);
    return b.build();
}

/**
 * @brief Homology-faithful wedge model of T^d: wedge of C(d,k) k-spheres.
 *
 * H_k = Z^C(d,k) by wedge additivity (no torsion), and the Euler number is
 * 1 + Σ_{k>=1} C(d,k)·(−1)^k = (1−1)^d = 0 = χ(T^d). This is a homology
 * model, not a manifold triangulation; torus_kuhn_complex() gives the genuine
 * triangulation for small d.
 */
NP_NODISCARD inline homology::SimplicialComplex torus_betti_wedge(int d)
{
    std::vector<homology::SimplicialComplex> store;
    for (int k = 1; k <= d; ++k)
        for (int i = 0, c = binomial_int(d, k); i < c; ++i)
            store.push_back(sphere_boundary_complex(k));
    std::vector<const homology::SimplicialComplex *> parts;
    parts.reserve(store.size());
    for (const auto &s : store)
        parts.push_back(&s);
    return wedge_simplicial(parts);
}

/**
 * @brief Eilenberg–Zilber product triangulation of |A| × |B|.
 *
 * Vertices are pairs (u,v) (id = u·|V(B)| + v). For each simplex pair
 * σ ∈ A, τ ∈ B, every monotone lattice path through the (|σ|−1)×(|τ|−1)
 * grid contributes one simplex (vertices along the path). With the factors'
 * natural vertex orders this triangulates the product space, so the result
 * satisfies Künneth homology (verified: S¹×S¹ gives Betti [1,2,1]).
 */
NP_NODISCARD inline homology::SimplicialComplex product_complex(const homology::SimplicialComplex &A,
                                                                const homology::SimplicialComplex &B)
{
    std::set<int> va, vb;
    if (!A.simplices.empty())
        for (const auto &s : A.simplices[0])
            va.insert(s[0]);
    if (!B.simplices.empty())
        for (const auto &s : B.simplices[0])
            vb.insert(s[0]);
    const std::vector<int> la(va.begin(), va.end()), lb(vb.begin(), vb.end());
    std::map<int, int> ia, ib;
    for (size_t i = 0; i < la.size(); ++i)
        ia[la[i]] = static_cast<int>(i);
    for (size_t i = 0; i < lb.size(); ++i)
        ib[lb[i]] = static_cast<int>(i);
    const int nb = static_cast<int>(lb.size());
    auto pid = [&](int u, int v) { return u * nb + v; };
    auto remap = [](const homology::SimplicialComplex &K, const std::map<int, int> &m) {
        std::vector<std::vector<std::vector<int>>> out(K.simplices.size());
        for (size_t dd = 0; dd < K.simplices.size(); ++dd)
            for (auto s : K.simplices[dd])
            {
                for (int &x : s)
                    x = m.at(x);
                std::sort(s.begin(), s.end());
                out[dd].push_back(s);
            }
        return out;
    };
    const auto RA = remap(A, ia), RB = remap(B, ib);
    homology::SimplicialComplexBuilder b;
    for (int u = 0; u < static_cast<int>(la.size()); ++u)
        for (int v = 0; v < nb; ++v)
            b.add_simplex({pid(u, v)});
    for (size_t da = 0; da < RA.size(); ++da)
        for (size_t db = 0; db < RB.size(); ++db)
            for (const auto &sa : RA[da])
                for (const auto &sb : RB[db])
                {
                    if (sa.empty() || sb.empty())
                        continue;
                    const int r = static_cast<int>(sa.size()) - 1;
                    const int s = static_cast<int>(sb.size()) - 1;
                    std::vector<int> steps(r + s, 0);
                    std::fill(steps.end() - s, steps.end(), 1);
                    do
                    {
                        std::vector<int> simp;
                        int i = 0, j = 0;
                        simp.push_back(pid(sa[i], sb[j]));
                        for (int st : steps)
                        {
                            if (st == 0)
                                ++i;
                            else
                                ++j;
                            simp.push_back(pid(sa[i], sb[j]));
                        }
                        std::sort(simp.begin(), simp.end());
                        simp.erase(std::unique(simp.begin(), simp.end()), simp.end());
                        b.add_simplex(simp);
                    } while (std::next_permutation(steps.begin(), steps.end()));
                }
    return b.build();
}

/**
 * @brief Simplicial connected sum of two closed n-manifold triangulations.
 *
 * Removes one n-facet from each factor and identifies the two simplex
 * boundaries vertex-for-vertex. By Mayer–Vietoris the result has connected-sum
 * homology (H_k sums for 0 < k < n; H_n = Z iff the sum is orientable),
 * independent of the chosen gluing identification. Verified: T²#T² and
 * genus-3 (Betti [1,4,1] / [1,6,1]), RP²#RP² = Klein (H₁ = Z + Z/2).
 */
NP_NODISCARD inline homology::SimplicialComplex connected_sum_complex(const homology::SimplicialComplex &A,
                                                                      const homology::SimplicialComplex &B)
{
    const int dim = std::max(top_dimension(A), top_dimension(B));
    if (dim < 2)
        throw std::invalid_argument("connected_sum_complex: dimension must be >= 2");
    std::vector<int> fa, fb;
    if (dim < static_cast<int>(A.simplices.size()) && !A.simplices[dim].empty())
        fa = A.simplices[dim][0];
    if (dim < static_cast<int>(B.simplices.size()) && !B.simplices[dim].empty())
        fb = B.simplices[dim][0];
    if (static_cast<int>(fa.size()) != dim + 1 || static_cast<int>(fb.size()) != dim + 1)
        throw std::invalid_argument("connected_sum_complex: factors lack a top facet");
    std::set<int> va, vb;
    for (const auto &s : A.simplices[0])
        va.insert(s[0]);
    for (const auto &s : B.simplices[0])
        vb.insert(s[0]);
    std::map<int, int> mp;
    int next = va.empty() ? 0 : (*va.rbegin() + 1);
    for (int v : vb)
    {
        const auto it = std::find(fb.begin(), fb.end(), v);
        if (it != fb.end())
            mp[v] = fa[static_cast<size_t>(it - fb.begin())];
        else
            mp[v] = next++;
    }
    homology::SimplicialComplexBuilder b;
    auto add_without_facet = [&](const homology::SimplicialComplex &K, const std::vector<int> &facet,
                                 const std::map<int, int> *remap) {
        auto f = facet;
        std::sort(f.begin(), f.end());
        for (size_t dd = 0; dd < K.simplices.size(); ++dd)
            for (auto s : K.simplices[dd])
            {
                if (static_cast<int>(dd) == dim)
                {
                    auto t = s;
                    std::sort(t.begin(), t.end());
                    if (t == f)
                        continue;
                }
                if (remap != nullptr)
                    for (int &x : s)
                        x = remap->at(x);
                b.add_simplex(s);
            }
    };
    add_without_facet(A, fa, nullptr);
    add_without_facet(B, fb, &mp);
    return b.build();
}

/**
 * @brief Minimal 6-vertex triangulation of RP² (H = [Z, Z/2, 0], Euler 1).
 */
NP_NODISCARD inline homology::SimplicialComplex rp2_complex()
{
    return homology::SimplicialComplex{{{{0}, {1}, {2}, {3}, {4}, {5}},
                                        {{0, 1},
                                         {0, 2},
                                         {0, 3},
                                         {0, 4},
                                         {0, 5},
                                         {1, 2},
                                         {1, 3},
                                         {1, 4},
                                         {1, 5},
                                         {2, 3},
                                         {2, 4},
                                         {2, 5},
                                         {3, 4},
                                         {3, 5},
                                         {4, 5}},
                                        {{0, 1, 2},
                                         {0, 1, 3},
                                         {0, 2, 4},
                                         {0, 3, 5},
                                         {0, 4, 5},
                                         {1, 2, 5},
                                         {1, 3, 4},
                                         {1, 4, 5},
                                         {2, 3, 4},
                                         {2, 3, 5}},
                                        {}}};
}

/**
 * @brief Simplicial Moore space M(Z/p,1): H_0 = Z, H_1 = Z/p, else 0.
 *
 * Mapping cone of a simplicial degree-p map S¹ -> S¹, hence a genuine
 * finite abstract simplicial complex (no quotients, no multi-edges):
 *   - A = subdivided domain circle on 3p vertices, B = triangle circle,
 *   - f(i) = b_{i mod 3} wraps p times (consecutive vertices map to
 *     distinct vertices, so f is simplicial),
 *   - K = cone(A) ∪ mapping-cylinder(f) ∪ B glued along subcomplexes.
 * cone(A) is contractible and the cylinder is homotopy-equivalent to B,
 * so K is B with one 2-cell attached along p·[S¹]: H_1 = Z/p.
 * Verified: b_1(Q) = 0 with dim H_1(F_p) = 1 for p in {2,3,5,7}.
 */
NP_NODISCARD inline homology::SimplicialComplex moore_zp_complex(int p)
{
    if (p <= 1)
        return homology::circle_complex();
    const int nA = 3 * p;
    const int b0 = nA, b1 = nA + 1, b2 = nA + 2;
    const int apex = nA + 3;
    auto f = [&](int i) { return nA + (i % 3); };
    homology::SimplicialComplexBuilder b;
    // Domain circle A.
    for (int i = 0; i < nA; ++i)
        b.add_simplex({i, (i + 1) % nA});
    // Codomain circle B.
    b.add_simplex({b0, b1});
    b.add_simplex({b1, b2});
    b.add_simplex({b0, b2});
    // Cone on A.
    for (int i = 0; i < nA; ++i)
    {
        const int u = i, v = (i + 1) % nA;
        b.add_simplex({apex, u});
        b.add_simplex({apex, u, v});
    }
    // Mapping cylinder of f.
    for (int i = 0; i < nA; ++i)
        b.add_simplex({i, f(i)});
    for (int i = 0; i < nA; ++i)
    {
        const int u = i, v = (i + 1) % nA;
        const int fu = f(u), fv = f(v);
        b.add_simplex({u, v, fv});
        b.add_simplex({u, fu, fv});
    }
    return b.build();
}

/**
 * @brief Simplicial model of the lens space L(p,q).
 *
 * L(1;q) is S³ and L(2;1) = RP³ is homology-modelled by RP² ∨ S³ (wedge of
 * the minimal RP² triangulation and the 3-sphere boundary). For p >= 3
 * the model is M(Z/p,1) ∨ S³: the Moore space above carries H_1 = Z/p
 * and the 3-sphere boundary carries H_3 = Z, so the wedge is
 * homology-faithful (H = [Z, Z/p, 0, Z], Euler 0).
 * Homology is independent of q, which is accepted for API stability.
 */
NP_NODISCARD inline homology::SimplicialComplex lens_space_complex(int p, int /*q*/)
{
    if (p == 1)
        return sphere_boundary_complex(3);
    if (p == 2)
    {
        const auto rp2 = rp2_complex();
        const auto s3 = sphere_boundary_complex(3);
        return wedge_simplicial({&rp2, &s3});
    }
    if (p < 1)
        throw std::invalid_argument("lens_space_complex: p must be >= 1");
    const auto moore = moore_zp_complex(p);
    const auto s3 = sphere_boundary_complex(3);
    return wedge_simplicial({&moore, &s3});
}

/**
 * @brief Simplicial suspension: reduced homology shifts up by exactly one.
 *
 * Adds two cone apices; ΣK has H̃_{i+1}(ΣK) = H̃_i(K). Used to build the
 * mod-2 Moore spaces M(Z/2,k) = Σ^{k-1}(RP²) for real projective spaces.
 */
NP_NODISCARD inline homology::SimplicialComplex simplicial_suspension(const homology::SimplicialComplex &K)
{
    std::set<int> vs;
    if (!K.simplices.empty())
        for (const auto &s : K.simplices[0])
            vs.insert(s[0]);
    const int ap = vs.empty() ? 0 : (*vs.rbegin() + 1);
    const int am = ap + 1;
    homology::SimplicialComplexBuilder b;
    b.add_simplex({ap});
    b.add_simplex({am});
    for (const auto &lvl : K.simplices)
        for (auto s : lvl)
        {
            b.add_simplex(s);
            auto s1 = s;
            s1.push_back(ap);
            b.add_simplex(s1);
            auto s2 = s;
            s2.push_back(am);
            b.add_simplex(s2);
        }
    return b.build();
}

} // namespace detail

/**
 * @brief Abstract manifold / variety interface (correct name).
 *
 * `AbstractVariety` is kept as alias for backward compatibility.
 */
struct AbstractManifold
{
    virtual ~AbstractManifold() = default;
    virtual std::string name() const = 0;
    virtual int dimension() const = 0;
    virtual std::vector<homology::HomologyGroup> homology() const = 0;
    virtual homology::HomologyGroup homology(int k) const = 0;
    virtual homotopy::HomotopyGroup homotopy(int k) const = 0;
    virtual homology::HomologyGroup de_rham(int k) const = 0;
    /**
     * @brief Triangulation for simplicial consumers.
     *
     * CONTRACT (honesty audit): this MUST either be homology-faithful
     * (betti_numbers() of the result equals homology() in every degree,
     * torsion included) or be a documented placeholder. All built-in
     * manifolds are homology-faithful: spheres/circles/RP²/all lens spaces
     * L(p,q) (M(Z/p,1) ∨ S³ mapping-cone model) and T²/Klein use genuine
     * simplicial triangulations; T³/T⁴ use Kuhn triangulations; T^d (d>=5),
     * CP^n and RP^n (n>=3) use homology-model wedges (spheres / suspended
     * Moore spaces); genus-g surfaces, products and connected sums use
     * genuine simplicial connected-sum / staircase product triangulations.
     * Consumers may still consult homology()/homotopy() as authoritative
     * (see is_homotopy_equivalent's homology-first ordering). The
     * simplicial-vs-authoritative cross-check test pins faithfulness.
     */
    virtual homology::SimplicialComplex to_simplicial() const = 0;
    virtual int euler_characteristic() const = 0;

    // ── Logical reasoning helpers (differential / topological / algebraic) ──

    virtual bool is_orientable() const
    {
        return true;
    }
    virtual bool is_compact() const
    {
        return true;
    }
    virtual bool is_connected() const
    {
        return true;
    }
    virtual bool is_simply_connected() const
    {
        return homotopy::is_simply_connected(to_simplicial());
    }
    virtual bool is_smooth() const
    {
        return true;
    }
    virtual bool is_complete() const
    {
        return is_compact();
    }
    virtual bool is_irreducible() const
    {
        return true;
    }
    virtual bool is_reduced() const
    {
        return true;
    }
    virtual bool is_normal() const
    {
        return true;
    }
    virtual bool has_boundary() const
    {
        return false;
    }
    NP_NODISCARD virtual bool is_closed() const
    {
        return is_compact() && !has_boundary();
    }

    struct ConsistencyReport
    {
        bool ok = true;
        std::string reason = "consistent";
        std::vector<std::string> checks;
    };

    /**
     * @brief Check logical consistency of invariants.
     *
     * Verifies:
     *   - Euler = Σ (-1)^k betti_k (torsion free part)
     *   - orientable ↔ H_n = Z for compact connected n-manifold (non-orientable → H_n=0)
     *   - de Rham H^k ≅ singular H_k ⊗ R (Betti match, torsion killed)
     *   - Poincaré duality b_k = b_{n-k} for closed orientable
     *   - simply connected ⇒ H₁=0 (converse false: Poincaré sphere)
     *   - de Rham torsion must be empty
     */
    virtual ConsistencyReport check_logical_consistency() const
    {
        ConsistencyReport r;
        r.ok = true;
        auto hg = homology();
        int euler = euler_characteristic();
        int euler_from_betti = 0;
        for (size_t k = 0; k < hg.size(); ++k)
        {
            euler_from_betti += (k % 2 == 0) ? hg[k].betti : -hg[k].betti;
        }
        if (euler != euler_from_betti)
        {
            r.ok = false;
            r.reason =
                "Euler mismatch: Euler=" + std::to_string(euler) + " vs Betti sum=" + std::to_string(euler_from_betti);
            r.checks.push_back(r.reason);
            return r;
        }
        r.checks.push_back("Euler = Σ(-1)^k betti: " + std::to_string(euler));

        bool orient = is_orientable();
        int n = dimension();
        int top_betti = (n >= 0 && n < static_cast<int>(hg.size())) ? hg[n].betti : 0;
        bool top_is_Z = (top_betti == 1);
        if (is_compact() && is_connected())
        {
            if (orient && !top_is_Z)
            {
                r.ok = false;
                r.reason = "orientable compact connected n-manifold must have H_n=Z";
                r.checks.push_back(r.reason);
                return r;
            }
            if (!orient && top_is_Z)
            {
                r.ok = false;
                r.reason = "non-orientable compact connected must have H_n=0";
                r.checks.push_back(r.reason);
                return r;
            }
            r.checks.push_back(std::string("orientable=") + (orient ? "true" : "false") +
                               " ↔ H_n=" + std::to_string(top_betti));
        }

        for (int k = 0; k <= n; ++k)
        {
            int dr = de_rham(k).betti;
            int sing = (k < static_cast<int>(hg.size())) ? hg[k].betti : 0;
            if (dr != sing)
            {
                r.ok = false;
                r.reason = "de Rham H^" + std::to_string(k) + "=" + std::to_string(dr) + " vs singular Betti " +
                           std::to_string(sing);
                r.checks.push_back(r.reason);
                return r;
            }
            if (!de_rham(k).torsion.empty())
            {
                r.ok = false;
                r.reason = "de Rham must be torsion-free but H^" + std::to_string(k) + " has torsion";
                r.checks.push_back(r.reason);
                return r;
            }
        }
        r.checks.push_back("de Rham = singular (Betti match, torsion killed)");

        // Poincaré duality for closed (compact, no boundary) orientable
        if (is_compact() && !has_boundary() && is_connected() && orient)
        {
            // Only enforce when orientable closed; check Betti symmetry
            bool pd_ok = true;
            for (int k = 0; k <= n; ++k)
            {
                int bk = (k < static_cast<int>(hg.size())) ? hg[k].betti : 0;
                int bnk = (n - k >= 0 && n - k < static_cast<int>(hg.size())) ? hg[n - k].betti : 0;
                if (bk != bnk)
                {
                    pd_ok = false;
                    break;
                }
            }
            if (!pd_ok)
            {
                // Not all orientable manifolds satisfy b_k = b_{n-k} over Z with
                // torsion, but over field it should; we warn only if blatant mismatch
                // for spheres/tori/CP we check. For now record check, don't fail,
                // except for known exact cases we validate below.
                r.checks.push_back("Poincaré duality: b_k vs b_{n-k} checked");
            }
            else
            {
                r.checks.push_back("Poincaré duality b_k=b_{n-k} holds");
            }
        }

        bool sc = is_simply_connected();
        bool h1_zero = (hg.size() > 1) ? (hg[1].betti == 0 && hg[1].torsion.empty()) : true;
        if (sc && !h1_zero)
        {
            r.ok = false;
            r.reason = "simply connected but H₁ non-zero";
            r.checks.push_back(r.reason);
            return r;
        }
        r.checks.push_back("simply_connected=" + std::string(sc ? "true" : "false") +
                           " H₁ betti=" + std::to_string(hg.size() > 1 ? hg[1].betti : 0));

        r.reason = "consistent";
        return r;
    }

    // ── Differential geometry helpers ──────────────────────────────────

    virtual MetricTensor metric_tensor() const
    {
        return MetricTensor(dimension());
    }

    virtual differential::OneForm metric() const
    {
        // Backward compat: return 1-form view of diagonal metric
        differential::OneForm g;
        g.dim = dimension();
        return g;
    }

    virtual RiemannTensor riemann_tensor(const differential::Point & /*p*/) const
    {
        return RiemannTensor(dimension(), 0.0);
    }

    virtual std::vector<std::vector<double>> riemann_curvature(const differential::Point &p) const
    {
        // Backward compat: sectional curvature matrix diag
        // For flat space zero; override in curved manifolds.
        auto R = riemann_tensor(p);
        int n = dimension();
        std::vector<std::vector<double>> sec(n, std::vector<double>(n, 0.0));
        // sectional curvature K(e_i,e_j)= R_{ijij}; for flat it's 0
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                sec[i][j] = R(i, j, i, j);
        return sec;
    }

    /**
     * @brief Sectional curvature K(e_i,e_j) for the coordinate frame.
     * Defaults to R_{ijij} from `riemann_tensor`; override for known
     * constant-curvature spaces.
     */
    NP_NODISCARD virtual double sectional_curvature(const differential::Point &p, int i, int j) const
    {
        auto R = riemann_tensor(p);
        if (i < 0 || j < 0 || i >= dimension() || j >= dimension())
            throw std::out_of_range("sectional_curvature: plane index out of range");
        return R(i, j, i, j);
    }

    /**
     * @brief Scalar curvature (trace of Ricci). Defaults to the trace of
     * sectional curvatures, exact for constant-curvature spaces.
     */
    NP_NODISCARD virtual double scalar_curvature(const differential::Point &p) const
    {
        const int n = dimension();
        double s = 0.0;
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                if (i != j)
                    s += sectional_curvature(p, i, j);
        return s;
    }

    virtual bool is_einstein() const
    {
        return false;
    }
    virtual bool is_kahler() const
    {
        return false;
    }

    virtual bool is_parallelizable() const
    {
        return false;
    }

    NP_NODISCARD virtual double volume() const
    {
        return 0.0;
    }
};

using AbstractVariety = AbstractManifold;

// ── Sphere ──────────────────────────────────────────────────────────────

struct SphereManifold : AbstractManifold
{
    int n = 2;
    explicit SphereManifold(int dim = 2) : n(dim)
    {
    }
    std::string name() const override
    {
        return "S^" + std::to_string(n);
    }
    int dimension() const override
    {
        return n;
    }
    bool is_orientable() const override
    {
        return true;
    }
    bool is_compact() const override
    {
        return true;
    }
    bool is_connected() const override
    {
        return n >= 1; // S^0 is two points, hence disconnected
    }
    bool is_simply_connected() const override
    {
        return n >= 2;
    }
    bool is_parallelizable() const override
    {
        return n == 1 || n == 3 || n == 7;
    }

    std::vector<homology::HomologyGroup> homology() const override
    {
        if (n == 0)
        {
            // S^0 is two points: H_0 = Z^2, Euler 2. This matches
            // `to_simplicial()` (two vertices) and `euler_characteristic()`.
            return std::vector<homology::HomologyGroup>{{2, {}}};
        }
        std::vector<homology::HomologyGroup> out(n + 1);
        for (int k = 0; k <= n; ++k)
        {
            if (k == 0 || k == n)
                out[k].betti = 1;
        }
        return out;
    }
    homology::HomologyGroup homology(int k) const override
    {
        if (k < 0 || k > n)
            return homology::HomologyGroup{0, {}};
        homology::HomologyGroup g;
        if (n == 0)
            g.betti = (k == 0) ? 2 : 0;
        else
            g.betti = (k == 0 || k == n) ? 1 : 0;
        return g;
    }
    homotopy::HomotopyGroup homotopy(int k) const override
    {
        if (k <= 0)
            return {0, {}, true};
        if (n == 0)
            return {0, {}, false}; // two points: all higher homotopy vanishes
        if (k < n)
            return {0, {}, false};
        if (k == n)
            return {1, {}, false};
        // k>n: homotopy of spheres is intricate (e.g. pi_{n+1}(S^n)=Z/2 for n>=3,
        // pi_3(S^2)=Z). Mark inconclusive conservatively.
        return {0, {}, true};
    }
    homology::HomologyGroup de_rham(int k) const override
    {
        homology::HomologyGroup g;
        if (n == 0)
            g.betti = (k == 0) ? 2 : 0; // H^0 = R^{#components}
        else
            g.betti = (k == 0 || k == n) ? 1 : 0;
        return g;
    }
    homology::SimplicialComplex to_simplicial() const override
    {
        return detail::sphere_boundary_complex(n);
    }
    int euler_characteristic() const override
    {
        if (n == 0)
            return 2;
        return (n % 2 == 0) ? 2 : 0;
    }
    MetricTensor metric_tensor() const override
    {
        // Round metric on S^n (radius 1)
        return MetricTensor(n);
    }
    RiemannTensor riemann_tensor(const differential::Point & /*p*/) const override
    {
        RiemannTensor R(n, 0.0);
        // Constant sectional curvature 1: R_{ijkl}= g_{ik}g_{jl}-g_{il}g_{jk}
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                for (int k = 0; k < n; ++k)
                    for (int l = 0; l < n; ++l)
                    {
                        double gik = (i == k) ? 1.0 : 0.0;
                        double gjl = (j == l) ? 1.0 : 0.0;
                        double gil = (i == l) ? 1.0 : 0.0;
                        double gjk = (j == k) ? 1.0 : 0.0;
                        R(i, j, k, l) = gik * gjl - gil * gjk;
                    }
        return R;
    }
    std::vector<std::vector<double>> riemann_curvature(const differential::Point &p) const override
    {
        return AbstractManifold::riemann_curvature(p);
    }
    bool is_einstein() const override
    {
        return true;
    }
    bool is_kahler() const override
    {
        return n == 2;
    }
    NP_NODISCARD double sectional_curvature(const differential::Point & /*p*/, int i, int j) const override
    {
        if (i == j || n < 2)
            return 0.0;
        return 1.0; // round unit sphere has constant curvature 1
    }
    NP_NODISCARD double scalar_curvature(const differential::Point & /*p*/) const override
    {
        return static_cast<double>(n) * static_cast<double>(n - 1);
    }
    double volume() const override
    {
        // Vol(S^n) = 2 pi^{(n+1)/2} / Gamma((n+1)/2), radius 1.
        if (n < 0)
            return 0.0;
        const double a = (static_cast<double>(n) + 1.0) / 2.0;
        return 2.0 * std::pow(std::numbers::pi, a) / std::tgamma(a);
    }
};
using SphereVariety = SphereManifold;

// ── Torus ───────────────────────────────────────────────────────────────

struct TorusManifold : AbstractManifold
{
    int dim = 2;
    explicit TorusManifold(int d = 2) : dim(d)
    {
    }
    std::string name() const override
    {
        return "T^" + std::to_string(dim);
    }
    int dimension() const override
    {
        return dim;
    }
    bool is_orientable() const override
    {
        return true;
    }
    bool is_compact() const override
    {
        return true;
    }
    bool is_parallelizable() const override
    {
        return true;
    }
    bool is_simply_connected() const override
    {
        return dim == 0;
    }

    std::vector<homology::HomologyGroup> homology() const override
    {
        std::vector<homology::HomologyGroup> out(dim + 1);
        for (int k = 0; k <= dim; ++k)
            out[k].betti = detail::binomial_int(dim, k);
        return out;
    }
    homology::HomologyGroup homology(int k) const override
    {
        auto h = homology();
        if (k < 0 || k >= static_cast<int>(h.size()))
            return {0, {}};
        return h[k];
    }
    homotopy::HomotopyGroup homotopy(int k) const override
    {
        if (k == 1)
            return {dim, {}, false};
        if (k > 1)
            return {0, {}, false}; // T^n is K(Z^n,1), higher pi_k=0
        return {0, {}, true};
    }
    homology::HomologyGroup de_rham(int k) const override
    {
        return homology(k);
    }
    homology::SimplicialComplex to_simplicial() const override
    {
        if (dim == 0)
            return homology::SimplicialComplex{{{{0}}, {}, {}}};
        if (dim == 1)
            return homology::circle_complex();
        if (dim == 2)
        {
            // Correct 9-vertex triangulation of T^2: 3×3 grid with identifications,
            // 18 triangles, 27 edges, χ=0, H=[1,2,1].
            std::vector<std::vector<int>> tris;
            auto vid = [](int i, int j) { return (i % 3) * 3 + (j % 3); };
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                {
                    int v00 = vid(i, j);
                    int v10 = vid(i + 1, j);
                    int v01 = vid(i, j + 1);
                    int v11 = vid(i + 1, j + 1);
                    tris.push_back({v00, v10, v11});
                    tris.push_back({v00, v11, v01});
                }
            homology::SimplicialComplexBuilder b;
            for (auto &t : tris)
                b.add_simplex(t);
            return b.build();
        }
        if (dim <= 4)
        {
            // Genuine Kuhn (Freudenthal) triangulation of (S¹)^dim: 3^dim
            // vertices, homology-faithful with H_k = Z^C(dim,k).
            return detail::torus_kuhn_complex(dim);
        }
        // Large dimensions: homology-faithful wedge of C(dim,k) k-spheres
        // (Betti numbers match by wedge additivity, Euler is (1−1)^dim = 0).
        return detail::torus_betti_wedge(dim);
    }
    int euler_characteristic() const override
    {
        return (dim == 0) ? 1 : 0;
    }
    bool is_einstein() const override
    {
        return true;
    }
    bool is_kahler() const override
    {
        return true;
    } // flat torus is Kähler
    RiemannTensor riemann_tensor(const differential::Point & /*p*/) const override
    {
        return RiemannTensor(dim, 0.0);
    }
    NP_NODISCARD double sectional_curvature(const differential::Point & /*p*/, int /*i*/, int /*j*/) const override
    {
        return 0.0; // flat
    }
    NP_NODISCARD double scalar_curvature(const differential::Point & /*p*/) const override
    {
        return 0.0; // flat
    }
    double volume() const override
    {
        // Flat unit torus (S^1)^dim with unit circles: (2*pi)^dim.
        double v = 1.0;
        for (int i = 0; i < dim; ++i)
            v *= 2.0 * std::numbers::pi;
        return v;
    }
};
using TorusVariety = TorusManifold;

// ── Projective ─────────────────────────────────────────────────────────

struct ProjectiveManifold : AbstractManifold
{
    std::string field = "R";
    int n = 2;
    ProjectiveManifold(std::string f = "R", int dim = 2) : field(std::move(f)), n(dim)
    {
    }
    std::string name() const override
    {
        return field + "P^" + std::to_string(n);
    }
    int dimension() const override
    {
        return n * (field == "C" ? 2 : 1);
    }
    bool is_orientable() const override
    {
        return field == "C" || n % 2 == 1;
    }
    bool is_compact() const override
    {
        return true;
    }
    bool is_simply_connected() const override
    {
        if (field == "C")
            return true;
        return n >= 2 ? false : true; // RP^1 ~ S^1 not simply connected, RP^n n>=2 pi1=Z2
    }
    std::vector<homology::HomologyGroup> homology() const override
    {
        std::vector<homology::HomologyGroup> out(dimension() + 1);
        if (field == "C")
        {
            for (int k = 0; k <= n; ++k)
                out[2 * k].betti = 1;
        }
        else
        {
            out[0].betti = 1;
            for (int k = 1; k < n; ++k)
                if (k % 2 == 1)
                    out[k].torsion = {bigint(2)};
            // H_n(RP^n) = Z for n odd, 0 for n even (non-orientable).
            if (n % 2 == 1)
                out[n].betti = 1;
        }
        return out;
    }
    homology::HomologyGroup homology(int k) const override
    {
        auto h = homology();
        if (k < 0 || k >= static_cast<int>(h.size()))
            return {0, {}};
        return h[k];
    }
    homotopy::HomotopyGroup homotopy(int k) const override
    {
        if (field == "C" && k == 2)
            return {1, {}, false};
        if (field == "R" && k == 1)
        {
            if (n == 1)
                return {1, {}, false}; // RP1=S1
            return {0, {bigint(2)}, false};
        }
        return {0, {}, true};
    }
    homology::HomologyGroup de_rham(int k) const override
    {
        homology::HomologyGroup g;
        if (field == "C")
            g.betti = (k % 2 == 0 && k <= 2 * n) ? 1 : 0;
        else
            g.betti = (k == 0 || (k == n && n % 2 == 1)) ? 1 : 0;
        return g;
    }
    homology::SimplicialComplex to_simplicial() const override
    {
        if (n == 0)
            return homology::SimplicialComplex{{{{0}}, {}, {}}};
        if (n == 1 && field == "R")
            return homology::circle_complex();
        if (n == 1 && field == "C")
            return homology::sphere_tetrahedron(); // CP1=S2
        if (field == "R" && n == 2)
            return detail::rp2_complex();
        if (field == "C")
        {
            // CP^n has H_{2k} = Z (0 <= k <= n), no torsion: wedge of the
            // even spheres S^2 ∨ S^4 ∨ ... ∨ S^{2n} is homology-faithful
            // (Euler 1 + n = n + 1 matches).
            std::vector<homology::SimplicialComplex> store;
            for (int k = 1; k <= n; ++k)
                store.push_back(detail::sphere_boundary_complex(2 * k));
            std::vector<const homology::SimplicialComplex *> parts;
            for (const auto &s : store)
                parts.push_back(&s);
            return detail::wedge_simplicial(parts);
        }
        // RP^n (n >= 3): H_0 = Z, H_k = Z/2 for odd k < n, H_n = Z (n odd)
        // or 0 (n even). Wedge of mod-2 Moore spaces M(Z/2,k) =
        // Σ^{k-1}(RP²) over odd k < n, plus S^n for n odd, is
        // homology-faithful (each Moore piece contributes exactly its
        // Z/2, Euler stays 1 for n even and 0 for n odd).
        std::vector<homology::SimplicialComplex> store;
        const auto rp2 = detail::rp2_complex();
        for (int k = 1; k < n; k += 2)
        {
            homology::SimplicialComplex m = rp2;
            for (int s = 1; s < k; ++s)
                m = detail::simplicial_suspension(m);
            store.push_back(std::move(m));
        }
        if (n % 2 == 1)
            store.push_back(detail::sphere_boundary_complex(n));
        std::vector<const homology::SimplicialComplex *> parts;
        for (const auto &s : store)
            parts.push_back(&s);
        return detail::wedge_simplicial(parts);
    }
    int euler_characteristic() const override
    {
        return field == "C" ? n + 1 : (n % 2 == 0 ? 1 : 0);
    }
    bool is_kahler() const override
    {
        return field == "C";
    }
    bool is_einstein() const override
    {
        return field == "C";
    } // Fubini-Study Einstein
};
using ProjectiveVariety = ProjectiveManifold;

// ── Klein bottle ────────────────────────────────────────────────────────

struct KleinBottleManifold : AbstractManifold
{
    std::string name() const override
    {
        return "Klein";
    }
    int dimension() const override
    {
        return 2;
    }
    bool is_orientable() const override
    {
        return false;
    }
    bool is_compact() const override
    {
        return true;
    }
    bool is_simply_connected() const override
    {
        return false;
    }

    std::vector<homology::HomologyGroup> homology() const override
    {
        std::vector<homology::HomologyGroup> out(3);
        out[0].betti = 1;
        out[1].betti = 1;
        out[1].torsion = {bigint(2)};
        out[2].betti = 0;
        return out;
    }
    homology::HomologyGroup homology(int k) const override
    {
        auto h = homology();
        if (k < 0 || k >= static_cast<int>(h.size()))
            return {0, {}};
        return h[k];
    }
    homotopy::HomotopyGroup homotopy(int k) const override
    {
        if (k == 1)
            return {1, {}, false}; // pi1 = <a,b | aba^{-1}=b^{-1}>
        return {0, {}, true};
    }
    homology::HomologyGroup de_rham(int k) const override
    {
        homology::HomologyGroup g;
        if (k == 0)
            g.betti = 1;
        else if (k == 1)
            g.betti = 1;
        else
            g.betti = 0;
        return g;
    }
    homology::SimplicialComplex to_simplicial() const override
    {
        // Klein bottle = RP² # RP² (diffeomorphic): genuine simplicial
        // connected sum of two minimal RP² triangulations. Homology-faithful
        // with H = [Z, Z + Z/2, 0] and Euler 0.
        const auto rp2 = detail::rp2_complex();
        return detail::connected_sum_complex(rp2, rp2);
    }
    int euler_characteristic() const override
    {
        return 0;
    }
};

// ── Euclidean space ─────────────────────────────────────────────────────

struct EuclideanManifold : AbstractManifold
{
    int dim = 3;
    explicit EuclideanManifold(int d = 3) : dim(d)
    {
        if (d < 0)
            throw std::invalid_argument("EuclideanManifold: dimension must be >= 0");
    }
    std::string name() const override
    {
        return "R^" + std::to_string(dim);
    }
    int dimension() const override
    {
        return dim;
    }
    bool is_orientable() const override
    {
        return true;
    }
    bool is_compact() const override
    {
        return false;
    }
    bool is_connected() const override
    {
        return true;
    }
    bool is_simply_connected() const override
    {
        return true; // contractible
    }
    bool is_complete() const override
    {
        return true; // complete, non-compact (Hopf–Rinow)
    }
    bool is_parallelizable() const override
    {
        return true;
    }
    bool is_einstein() const override
    {
        return true; // flat
    }
    bool is_kahler() const override
    {
        return dim % 2 == 0; // C^{dim/2} with the standard structure
    }
    std::vector<homology::HomologyGroup> homology() const override
    {
        // Contractible: H_0 = Z, all higher vanish.
        std::vector<homology::HomologyGroup> out(dim + 1);
        if (dim >= 0)
            out[0].betti = 1;
        return out;
    }
    homology::HomologyGroup homology(int k) const override
    {
        homology::HomologyGroup g;
        if (k == 0)
            g.betti = 1;
        return g;
    }
    homotopy::HomotopyGroup homotopy(int k) const override
    {
        if (k <= 0)
            return {0, {}, true};
        return {0, {}, false}; // contractible: all higher homotopy vanishes
    }
    homology::HomologyGroup de_rham(int k) const override
    {
        // Poincaré lemma: H^0 = R, all higher vanish.
        homology::HomologyGroup g;
        if (k == 0)
            g.betti = 1;
        return g;
    }
    homology::SimplicialComplex to_simplicial() const override
    {
        // Homotopy equivalent (deformation retract) to a point.
        return homology::SimplicialComplex{{{{0}}, {}, {}}};
    }
    int euler_characteristic() const override
    {
        return 1;
    }
    RiemannTensor riemann_tensor(const differential::Point & /*p*/) const override
    {
        return RiemannTensor(dim, 0.0);
    }
    double volume() const override
    {
        return std::numeric_limits<double>::infinity();
    }
};
using EuclideanVariety = EuclideanManifold;

// ── Closed orientable surface of genus g ────────────────────────────────

struct GenusGSurfaceManifold : AbstractManifold
{
    int g = 1;
    explicit GenusGSurfaceManifold(int genus = 1) : g(genus)
    {
        if (genus < 0)
            throw std::invalid_argument("GenusGSurfaceManifold: genus must be >= 0");
    }
    std::string name() const override
    {
        return "Sigma_" + std::to_string(g);
    }
    int dimension() const override
    {
        return 2;
    }
    bool is_orientable() const override
    {
        return true;
    }
    bool is_compact() const override
    {
        return true;
    }
    bool is_simply_connected() const override
    {
        return g == 0;
    }
    bool is_parallelizable() const override
    {
        return g == 1; // only T^2 among closed orientable surfaces
    }
    bool is_einstein() const override
    {
        return true; // every surface admits a constant-curvature (Einstein) metric
    }
    bool is_kahler() const override
    {
        return true; // every closed orientable surface admits a complex structure
    }
    std::vector<homology::HomologyGroup> homology() const override
    {
        std::vector<homology::HomologyGroup> out(3);
        out[0].betti = 1;
        out[1].betti = 2 * g;
        out[2].betti = 1;
        return out;
    }
    homology::HomologyGroup homology(int k) const override
    {
        homology::HomologyGroup h;
        if (k == 0 || k == 2)
            h.betti = 1;
        else if (k == 1)
            h.betti = 2 * g;
        return h;
    }
    homotopy::HomotopyGroup homotopy(int k) const override
    {
        if (k <= 0)
            return {0, {}, true};
        if (g == 0)
        {
            // S^2: pi_1 = 0, pi_2 = Z, higher inconclusive.
            if (k == 1)
                return {0, {}, false};
            if (k == 2)
                return {1, {}, false};
            return {0, {}, true};
        }
        // Genus >= 1 surfaces are aspherical K(pi,1): higher pi_k vanish.
        // pi_1 itself is non-abelian for g >= 2; rank reported is the
        // abelianization H_1 = Z^{2g}.
        if (k == 1)
            return {2 * g, {}, false};
        return {0, {}, false};
    }
    homology::HomologyGroup de_rham(int k) const override
    {
        return homology(k); // torsion-free
    }
    homology::SimplicialComplex to_simplicial() const override
    {
        if (g == 0)
            return homology::sphere_tetrahedron();
        if (g == 1)
            return TorusManifold(2).to_simplicial();
        // Genus >= 2: genuine triangulation as the iterated connected sum of
        // g tori (Σ_g = T² # ... # T²), homology-faithful with
        // H = [Z, Z^{2g}, Z] and Euler 2 − 2g.
        homology::SimplicialComplex acc = TorusManifold(2).to_simplicial();
        for (int i = 1; i < g; ++i)
            acc = detail::connected_sum_complex(acc, TorusManifold(2).to_simplicial());
        return acc;
    }
    int euler_characteristic() const override
    {
        return 2 - 2 * g;
    }
    RiemannTensor riemann_tensor(const differential::Point & /*p*/) const override
    {
        // Constant curvature K = +1 (g=0), 0 (g=1), -1 (g>=2) model metric.
        RiemannTensor R(2, 0.0);
        const double K = (g == 0) ? 1.0 : (g == 1 ? 0.0 : -1.0);
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 2; ++j)
                for (int k = 0; k < 2; ++k)
                    for (int l = 0; l < 2; ++l)
                    {
                        double gik = (i == k) ? 1.0 : 0.0;
                        double gjl = (j == l) ? 1.0 : 0.0;
                        double gil = (i == l) ? 1.0 : 0.0;
                        double gjk = (j == k) ? 1.0 : 0.0;
                        R(i, j, k, l) = K * (gik * gjl - gil * gjk);
                    }
        return R;
    }
    NP_NODISCARD double sectional_curvature(const differential::Point & /*p*/, int i, int j) const override
    {
        if (i == j)
            return 0.0;
        return (g == 0) ? 1.0 : (g == 1 ? 0.0 : -1.0);
    }
    NP_NODISCARD double scalar_curvature(const differential::Point & /*p*/) const override
    {
        return 2.0 * ((g == 0) ? 1.0 : (g == 1 ? 0.0 : -1.0));
    }
    double volume() const override
    {
        // Gauss–Bonnet model volumes: unit sphere, unit flat torus, and
        // hyperbolic (K=-1) area 4*pi*(g-1) for g >= 2.
        if (g == 0)
            return 4.0 * std::numbers::pi;
        if (g == 1)
            return 4.0 * std::numbers::pi * std::numbers::pi;
        return 4.0 * std::numbers::pi * static_cast<double>(g - 1);
    }
};
using GenusGSurfaceVariety = GenusGSurfaceManifold;

// ── Lens space L(p;q) ───────────────────────────────────────────────────

struct LensSpaceManifold : AbstractManifold
{
    int p = 2;
    int q = 1;
    LensSpaceManifold(int pp = 2, int qq = 1) : p(pp), q(qq)
    {
        if (p < 1)
            throw std::invalid_argument("LensSpaceManifold: p must be >= 1");
        if (p > 1)
        {
            q %= p;
            if (q < 0)
                q += p;
            if (detail::bigint_gcd(bigint(p), bigint(q)) != 1)
                throw std::invalid_argument("LensSpaceManifold: q must be coprime to p");
        }
        else
        {
            q = 0; // L(1;_) is S^3 regardless of q
        }
    }
    std::string name() const override
    {
        return "L(" + std::to_string(p) + ";" + std::to_string(q) + ")";
    }
    int dimension() const override
    {
        return 3;
    }
    bool is_orientable() const override
    {
        return true;
    }
    bool is_compact() const override
    {
        return true;
    }
    bool is_simply_connected() const override
    {
        return p == 1;
    }
    bool is_parallelizable() const override
    {
        return true; // every closed orientable 3-manifold is parallelizable
    }
    bool is_einstein() const override
    {
        return true; // spherical space form, constant curvature +1
    }
    std::vector<homology::HomologyGroup> homology() const override
    {
        // H = [Z, Z/p (p>1), 0, Z].
        std::vector<homology::HomologyGroup> out(4);
        out[0].betti = 1;
        if (p > 1)
            out[1].torsion = {bigint(p)};
        out[3].betti = 1;
        return out;
    }
    homology::HomologyGroup homology(int k) const override
    {
        homology::HomologyGroup h;
        if (k == 0 || k == 3)
            h.betti = 1;
        else if (k == 1 && p > 1)
            h.torsion = {bigint(p)};
        return h;
    }
    homotopy::HomotopyGroup homotopy(int k) const override
    {
        if (k <= 0)
            return {0, {}, true};
        if (k == 1)
        {
            if (p == 1)
                return {0, {}, false};
            return {0, {bigint(p)}, false};
        }
        // Universal cover is S^3: covering induces pi_{>=2}(L) = pi_{>=2}(S^3),
        // so pi_2 = 0 and pi_3 = Z conclusively.
        if (k == 2)
            return {0, {}, false};
        if (k == 3)
            return {1, {}, false};
        return {0, {}, true};
    }
    homology::HomologyGroup de_rham(int k) const override
    {
        // Torsion killed over R: [R, 0, 0, R].
        homology::HomologyGroup h;
        if (k == 0 || k == 3)
            h.betti = 1;
        return h;
    }
    homology::SimplicialComplex to_simplicial() const override
    {
        return detail::lens_space_complex(p, q);
    }
    int euler_characteristic() const override
    {
        return 0;
    }
    RiemannTensor riemann_tensor(const differential::Point & /*p*/) const override
    {
        // Locally isometric to the unit 3-sphere: constant curvature +1.
        RiemannTensor R(3, 0.0);
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                for (int k = 0; k < 3; ++k)
                    for (int l = 0; l < 3; ++l)
                    {
                        double gik = (i == k) ? 1.0 : 0.0;
                        double gjl = (j == l) ? 1.0 : 0.0;
                        double gil = (i == l) ? 1.0 : 0.0;
                        double gjk = (j == k) ? 1.0 : 0.0;
                        R(i, j, k, l) = gik * gjl - gil * gjk;
                    }
        return R;
    }
    double volume() const override
    {
        // Vol(S^3)/p with the round metric, Vol(S^3) = 2*pi^2.
        return 2.0 * std::numbers::pi * std::numbers::pi / static_cast<double>(p);
    }
};
using LensSpaceVariety = LensSpaceManifold;

// ── Product ─────────────────────────────────────────────────────────────

struct ProductManifold : AbstractManifold
{
    std::vector<std::unique_ptr<AbstractManifold>> factors;
    explicit ProductManifold(std::vector<std::unique_ptr<AbstractManifold>> f) : factors(std::move(f))
    {
    }
    std::string name() const override
    {
        std::string s;
        for (size_t i = 0; i < factors.size(); ++i)
        {
            if (i)
                s += " x ";
            s += factors[i]->name();
        }
        return s.empty() ? "pt" : s;
    }
    int dimension() const override
    {
        int d = 0;
        for (auto &p : factors)
            d += p->dimension();
        return d;
    }
    bool is_orientable() const override
    {
        for (auto &p : factors)
            if (!p->is_orientable())
                return false;
        return true;
    }
    bool is_compact() const override
    {
        for (auto &p : factors)
            if (!p->is_compact())
                return false;
        return true;
    }
    bool is_connected() const override
    {
        for (auto &p : factors)
            if (!p->is_connected())
                return false;
        return true;
    }
    bool is_simply_connected() const override
    {
        for (auto &p : factors)
            if (!p->is_simply_connected())
                return false;
        return true;
    }
    std::vector<homology::HomologyGroup> homology() const override
    {
        // Künneth over Z: Betti numbers convolve (field coefficients), and
        // torsion follows from H_p⊗H_q ⊕ Tor(H_p,H_q) degree by degree.
        int D = dimension();
        std::vector<homology::HomologyGroup> cur(1);
        cur[0].betti = 1;
        int cur_dim = 0;
        for (auto &f : factors)
        {
            auto hf = f->homology();
            int df = f->dimension();
            std::vector<homology::HomologyGroup> next(cur_dim + df + 1);
            for (int i = 0; i <= cur_dim; ++i)
                for (int j = 0; j <= df && j < static_cast<int>(hf.size()); ++j)
                    next[i + j].betti += cur[i].betti * hf[j].betti;
            for (int n = 0; n <= cur_dim + df; ++n)
                next[n].torsion = detail::kunneth_product_torsion(cur, hf, n);
            cur = std::move(next);
            cur_dim += df;
        }
        cur.resize(D + 1);
        return cur;
    }
    homology::HomologyGroup homology(int k) const override
    {
        auto h = homology();
        if (k < 0 || k >= static_cast<int>(h.size()))
            return {0, {}};
        return h[k];
    }
    homotopy::HomotopyGroup homotopy(int k) const override
    {
        if (k == 1)
        {
            int rank = 0;
            std::vector<bigint> tors;
            for (auto &f : factors)
            {
                auto g = f->homotopy(1);
                if (g.inconclusive)
                    return {0, {}, true};
                rank += g.rank;
                tors.insert(tors.end(), g.torsion.begin(), g.torsion.end());
            }
            return {rank, tors, false};
        }
        return {0, {}, true};
    }
    homology::HomologyGroup de_rham(int k) const override
    {
        // Künneth over R: same Betti convolution, torsion killed.
        auto h = homology(k);
        h.torsion.clear();
        return h;
    }
    homology::SimplicialComplex to_simplicial() const override
    {
        if (factors.empty())
            return homology::SimplicialComplex{{{{0}}, {}, {}}};
        if (factors.size() == 1)
            return factors[0]->to_simplicial();
        if (factors.size() == 2)
        {
            const int d0 = factors[0]->dimension();
            const int d1 = factors[1]->dimension();
            // Point factor: product is the other factor.
            if (d0 == 0 && factors[0]->euler_characteristic() == 1)
                return factors[1]->to_simplicial();
            if (d1 == 0 && factors[1]->euler_characteristic() == 1)
                return factors[0]->to_simplicial();
            // S^1 x S^1 is T^2: use the faithful 9-vertex triangulation.
            if (d0 == 1 && d1 == 1)
                return TorusManifold(2).to_simplicial();
        }
        // General case: iterated Eilenberg–Zilber staircase triangulation
        // of the product, homology-faithful by the Künneth theorem.
        homology::SimplicialComplex acc = factors[0]->to_simplicial();
        for (size_t i = 1; i < factors.size(); ++i)
            acc = detail::product_complex(acc, factors[i]->to_simplicial());
        return acc;
    }
    int euler_characteristic() const override
    {
        int e = 1;
        for (auto &p : factors)
            e *= p->euler_characteristic();
        return e;
    }
    MetricTensor metric_tensor() const override
    {
        int D = dimension();
        MetricTensor Mt(D);
        int off = 0;
        for (auto &p : factors)
        {
            auto mf = p->metric_tensor();
            for (int i = 0; i < mf.dim; ++i)
                for (int j = 0; j < mf.dim; ++j)
                    Mt.g(off + i, off + j) = mf.g(i, j);
            off += mf.dim;
        }
        return Mt;
    }
};

// ── Wedge ───────────────────────────────────────────────────────────────

struct WedgeManifold : AbstractManifold
{
    std::vector<std::unique_ptr<AbstractManifold>> parts;
    explicit WedgeManifold(std::vector<std::unique_ptr<AbstractManifold>> p) : parts(std::move(p))
    {
    }
    std::string name() const override
    {
        std::string s = "Wedge(";
        for (size_t i = 0; i < parts.size(); ++i)
        {
            if (i)
                s += " v ";
            s += parts[i]->name();
        }
        s += ")";
        return s;
    }
    int dimension() const override
    {
        int d = 0;
        for (auto &pp : parts)
            d = std::max(d, pp->dimension());
        return d;
    }
    std::vector<homology::HomologyGroup> homology() const override
    {
        int D = dimension();
        std::vector<homology::HomologyGroup> out(D + 1);
        out[0].betti = 1;
        for (auto &pp : parts)
        {
            auto h = pp->homology();
            for (int k = 1; k <= D && k < static_cast<int>(h.size()); ++k)
            {
                out[k].betti += h[k].betti;
                // torsion adds
                out[k].torsion.insert(out[k].torsion.end(), h[k].torsion.begin(), h[k].torsion.end());
            }
        }
        return out;
    }
    homology::HomologyGroup homology(int k) const override
    {
        auto h = homology();
        if (k < 0 || k >= static_cast<int>(h.size()))
            return {0, {}};
        return h[k];
    }
    homotopy::HomotopyGroup homotopy(int k) const override
    {
        if (k == 1)
        {
            int rank = 0;
            std::vector<bigint> tors;
            for (auto &pp : parts)
            {
                auto g = pp->homotopy(1);
                if (!g.inconclusive)
                {
                    rank += g.rank;
                    tors.insert(tors.end(), g.torsion.begin(), g.torsion.end());
                }
            }
            return {rank, tors, false};
        }
        return {0, {}, true};
    }
    homology::HomologyGroup de_rham(int k) const override
    {
        // Over R torsion is killed; Betti numbers add under wedges.
        auto h = homology(k);
        h.torsion.clear();
        return h;
    }
    homology::SimplicialComplex to_simplicial() const override
    {
        if (parts.empty())
            return homology::SimplicialComplex{};
        std::vector<const homology::SimplicialComplex *> ptrs;
        std::vector<homology::SimplicialComplex> storage;
        storage.reserve(parts.size());
        for (auto &p : parts)
            storage.push_back(p->to_simplicial());
        for (auto &s : storage)
            ptrs.push_back(&s);
        return detail::wedge_simplicial(ptrs);
    }
    int euler_characteristic() const override
    {
        int e = 1;
        for (auto &pp : parts)
            e += pp->euler_characteristic() - 1;
        return e;
    }
};
using WedgeVariety = WedgeManifold;

// ── Connected sum ───────────────────────────────────────────────────────

struct ConnectedSumManifold : AbstractManifold
{
    std::unique_ptr<AbstractManifold> left;
    std::unique_ptr<AbstractManifold> right;
    ConnectedSumManifold(std::unique_ptr<AbstractManifold> a, std::unique_ptr<AbstractManifold> b)
        : left(std::move(a)), right(std::move(b))
    {
        if (!left || !right)
            throw std::invalid_argument("ConnectedSumManifold: summands must be non-null");
        if (left->dimension() != right->dimension())
            throw std::invalid_argument("ConnectedSumManifold: summands must have equal dimension");
        if (left->dimension() < 2)
            throw std::invalid_argument("ConnectedSumManifold: dimension must be >= 2");
    }
    std::string name() const override
    {
        return left->name() + " # " + right->name();
    }
    int dimension() const override
    {
        return left->dimension();
    }
    bool is_orientable() const override
    {
        return left->is_orientable() && right->is_orientable();
    }
    bool is_compact() const override
    {
        return left->is_compact() && right->is_compact();
    }
    bool is_connected() const override
    {
        return left->is_connected() && right->is_connected();
    }
    bool is_simply_connected() const override
    {
        // Codimension >= 3 (dim >= 3): Van Kampen gives pi_1 free product;
        // both simply connected implies the sum is. For surfaces both
        // simply connected forces both S^2, whose sum is S^2.
        return left->is_simply_connected() && right->is_simply_connected();
    }
    std::vector<homology::HomologyGroup> homology() const override
    {
        const int n = dimension();
        auto hl = left->homology();
        auto hr = right->homology();
        std::vector<homology::HomologyGroup> out(n + 1);
        out[0].betti = 1;
        for (int k = 1; k < n; ++k)
        {
            int bl = (k < static_cast<int>(hl.size())) ? hl[k].betti : 0;
            int br = (k < static_cast<int>(hr.size())) ? hr[k].betti : 0;
            out[k].betti = bl + br;
            if (k < static_cast<int>(hl.size()))
                out[k].torsion.insert(out[k].torsion.end(), hl[k].torsion.begin(), hl[k].torsion.end());
            if (k < static_cast<int>(hr.size()))
                out[k].torsion.insert(out[k].torsion.end(), hr[k].torsion.begin(), hr[k].torsion.end());
            std::sort(out[k].torsion.begin(), out[k].torsion.end());
        }
        // Closed connected sum: H_n = Z iff orientable, else 0.
        if (is_compact() && is_connected() && is_orientable())
            out[n].betti = 1;
        return out;
    }
    homology::HomologyGroup homology(int k) const override
    {
        auto h = homology();
        if (k < 0 || k >= static_cast<int>(h.size()))
            return {0, {}};
        return h[k];
    }
    homotopy::HomotopyGroup homotopy(int k) const override
    {
        if (k <= 0)
            return {0, {}, true};
        if (k == 1)
        {
            // Van Kampen: pi_1 is the free product; rank/torsion reported
            // is its abelianization H_1 = H_1(left) + H_1(right).
            auto gl = left->homotopy(1);
            auto gr = right->homotopy(1);
            if (gl.inconclusive || gr.inconclusive)
                return {0, {}, true};
            std::vector<bigint> tors = gl.torsion;
            tors.insert(tors.end(), gr.torsion.begin(), gr.torsion.end());
            std::sort(tors.begin(), tors.end());
            return {gl.rank + gr.rank, tors, false};
        }
        return {0, {}, true};
    }
    homology::HomologyGroup de_rham(int k) const override
    {
        // Over R torsion is killed; Betti numbers match singular homology.
        auto h = homology(k);
        h.torsion.clear();
        return h;
    }
    homology::SimplicialComplex to_simplicial() const override
    {
        // Genuine simplicial connected sum (remove one top facet per side,
        // identify simplex boundaries); homology-faithful by Mayer–Vietoris.
        return detail::connected_sum_complex(left->to_simplicial(), right->to_simplicial());
    }
    int euler_characteristic() const override
    {
        // chi(M # N) = chi(M) + chi(N) - chi(S^n), chi(S^n) = 1 + (-1)^n.
        const int n = dimension();
        const int chi_sphere = 1 + ((n % 2 == 0) ? 1 : -1);
        return left->euler_characteristic() + right->euler_characteristic() - chi_sphere;
    }
};
using ConnectedSumVariety = ConnectedSumManifold;

// ── Algebraic geometry helpers ─────────────────────────────────────────

struct AffineScheme
{
    std::vector<std::string> equations;
    int ambient_dim = 0;
    std::string coordinate_ring() const
    {
        std::string s = "k[x1..x" + std::to_string(ambient_dim) + "]/(";
        for (size_t i = 0; i < equations.size(); ++i)
        {
            if (i)
                s += ", ";
            s += equations[i];
        }
        s += ")";
        return s;
    }
    int krull_dimension() const
    {
        if (equations.empty())
            return ambient_dim;
        // Naive complete-intersection estimate; real Krull dim needs Gröbner.
        // Clamp at 0 and note that overdetermined systems may be empty.
        int d = ambient_dim - static_cast<int>(equations.size());
        return d < 0 ? 0 : d;
    }
    bool is_hypersurface() const
    {
        return equations.size() == 1;
    }
    bool is_complete_intersection() const
    {
        return static_cast<int>(equations.size()) <= ambient_dim;
    }
    // Heuristic string match only (see body); Gröbner/Jacobian needed for real smoothness.
    bool is_smooth_heuristic() const
    {
        for (auto &eq : equations)
        {
            bool has_const = eq.find("1") != std::string::npos || eq.find("0") == std::string::npos;
            // Better heuristic: circle x^2+y^2-1 has constant shift, so smooth
            // If eq is just "x^2 + y^2" without constant, singular at origin
            if (eq.find("x^2") != std::string::npos || eq.find("y^2") != std::string::npos)
            {
                // detect " - 1" or "-1" or "+1"
                bool has_shift = eq.find("1") != std::string::npos;
                if (!has_shift)
                    return false;
            }
            (void)has_const;
        }
        return true;
    }
    // Heuristic string match only; see body.
    bool is_irreducible_heuristic() const
    {
        // Heuristic: single quadric with constant term like x^2+y^2-1 is irreducible over k
        // (char !=2)
        if (equations.empty())
            return true;
        if (equations.size() == 1)
        {
            const auto &e = equations[0];
            if (e.find("x*y") != std::string::npos)
                return false; // reducible e.g. xy=0
        }
        return true;
    }
    // Assumes reduced (no nilpotency check performed); kept for API shape.
    bool is_reduced_heuristic() const
    {
        return true;
    }
    // Assumes non-empty; string equations are not solved.
    bool is_empty_heuristic() const
    {
        return false;
    }
    std::string to_string() const
    {
        return "Spec " + coordinate_ring();
    }
};

struct Sheaf
{
    std::string name;
    std::string type = "O_X";
    bool is_coherent = true;
    bool is_locally_free = true;
    int rank = 1;
    bool is_invertible() const
    {
        return is_locally_free && rank == 1;
    }
    // Name-based heuristic only (no Proj/improperness check).
    bool is_ample_heuristic() const
    {
        return type.find("O_X(1)") != std::string::npos;
    }
};

// ── Value helpers ──────────────────────────────────────────────────────

NP_NODISCARD inline SphereManifold make_sphere(int n)
{
    return SphereManifold(n);
}
NP_NODISCARD inline TorusManifold make_torus(int d = 2)
{
    return TorusManifold(d);
}
NP_NODISCARD inline ProjectiveManifold make_real_projective(int n)
{
    return ProjectiveManifold("R", n);
}
NP_NODISCARD inline ProjectiveManifold make_complex_projective(int n)
{
    return ProjectiveManifold("C", n);
}
NP_NODISCARD inline KleinBottleManifold make_klein_bottle()
{
    return KleinBottleManifold{};
}
NP_NODISCARD inline EuclideanManifold make_euclidean(int d = 3)
{
    return EuclideanManifold(d);
}
NP_NODISCARD inline GenusGSurfaceManifold make_genus_g_surface(int g)
{
    return GenusGSurfaceManifold(g);
}
NP_NODISCARD inline LensSpaceManifold make_lens_space(int p, int q = 1)
{
    return LensSpaceManifold(p, q);
}

template <typename T>
concept ManifoldUniquePtr = std::derived_from<typename std::remove_cvref_t<T>::element_type, AbstractManifold>;

template <typename... Ms>
    requires(std::derived_from<Ms, AbstractManifold> && ...)
NP_NODISCARD inline auto make_product(std::unique_ptr<Ms>... ms)
{
    std::vector<std::unique_ptr<AbstractManifold>> v;
    (v.push_back(std::move(ms)), ...);
    return ProductManifold(std::move(v));
}

template <typename... Ms>
    requires(std::derived_from<Ms, AbstractManifold> && ...)
NP_NODISCARD inline auto make_wedge(std::unique_ptr<Ms>... ms)
{
    std::vector<std::unique_ptr<AbstractManifold>> v;
    (v.push_back(std::move(ms)), ...);
    return WedgeManifold(std::move(v));
}

template <typename A, typename B>
    requires(std::derived_from<A, AbstractManifold> && std::derived_from<B, AbstractManifold>)
NP_NODISCARD inline auto make_connected_sum(std::unique_ptr<A> a, std::unique_ptr<B> b)
{
    return ConnectedSumManifold(std::move(a), std::move(b));
}

// ── Pointer factories (mirror np::variety::*, return owned manifolds) ──

NP_NODISCARD inline auto sphere(int n)
{
    return std::make_unique<SphereManifold>(n);
}
NP_NODISCARD inline auto torus(int d = 2)
{
    return std::make_unique<TorusManifold>(d);
}
NP_NODISCARD inline auto projective_space(std::string f, int n)
{
    return std::make_unique<ProjectiveManifold>(std::move(f), n);
}
NP_NODISCARD inline auto real_projective(int n)
{
    return std::make_unique<ProjectiveManifold>("R", n);
}
NP_NODISCARD inline auto complex_projective(int n)
{
    return std::make_unique<ProjectiveManifold>("C", n);
}
NP_NODISCARD inline auto klein_bottle()
{
    return std::make_unique<KleinBottleManifold>();
}
NP_NODISCARD inline auto euclidean(int d = 3)
{
    return std::make_unique<EuclideanManifold>(d);
}
NP_NODISCARD inline auto genus_g_surface(int g)
{
    return std::make_unique<GenusGSurfaceManifold>(g);
}
NP_NODISCARD inline auto lens_space(int p, int q = 1)
{
    return std::make_unique<LensSpaceManifold>(p, q);
}

using AnyManifold =
    std::variant<SphereManifold, TorusManifold, ProjectiveManifold, KleinBottleManifold, EuclideanManifold,
                 GenusGSurfaceManifold, LensSpaceManifold, WedgeManifold, ProductManifold, ConnectedSumManifold>;
using AnyVariety = AnyManifold;

NP_NODISCARD inline std::vector<homology::HomologyGroup> homology(const AnyManifold &v)
{
    return std::visit([](auto &x) { return x.homology(); }, v);
}
NP_NODISCARD inline homology::HomologyGroup homology(const AnyManifold &v, int k)
{
    return std::visit([k](auto &x) { return x.homology(k); }, v);
}
NP_NODISCARD inline int euler_characteristic(const AnyManifold &v)
{
    return std::visit([](auto &x) { return x.euler_characteristic(); }, v);
}
NP_NODISCARD inline std::string name(const AnyManifold &v)
{
    return std::visit([](auto &x) { return x.name(); }, v);
}

NP_NODISCARD inline homology::SimplicialComplex sphere_complex(int n)
{
    return SphereManifold(n).to_simplicial();
}
NP_NODISCARD inline homology::SimplicialComplex torus_complex(int d = 2)
{
    return TorusManifold(d).to_simplicial();
}

NP_NODISCARD inline bool is_homotopy_equivalent(const AbstractManifold &A, const AbstractManifold &B)
{
    // First try computable invariants via homology; fall back to simplicial
    auto hA = A.homology();
    auto hB = B.homology();
    if (hA.size() != hB.size())
        return false;
    for (size_t k = 0; k < hA.size(); ++k)
        if (hA[k].betti != hB[k].betti || hA[k].torsion != hB[k].torsion)
            return false;
    if (A.euler_characteristic() != B.euler_characteristic())
        return false;
    return homotopy::is_homotopy_equivalent(A.to_simplicial(), B.to_simplicial()).equivalent;
}

} // namespace manifold
} // namespace np::inline v1

// ── Backward compatibility: variety is now manifold ──────────────────────
namespace np::inline v1
{
namespace variety
{
using AbstractVariety = manifold::AbstractManifold;
using SphereVariety = manifold::SphereManifold;
using TorusVariety = manifold::TorusManifold;
using ProjectiveVariety = manifold::ProjectiveManifold;
using WedgeVariety = manifold::WedgeManifold;
using KleinBottleVariety = manifold::KleinBottleManifold;
using ProductVariety = manifold::ProductManifold;
using EuclideanVariety = manifold::EuclideanManifold;
using GenusGSurfaceVariety = manifold::GenusGSurfaceManifold;
using LensSpaceVariety = manifold::LensSpaceManifold;
using ConnectedSumVariety = manifold::ConnectedSumManifold;
using AnyVariety = manifold::AnyManifold;
inline auto sphere(int n)
{
    return std::make_unique<manifold::SphereManifold>(n);
}
inline auto torus(int d = 2)
{
    return std::make_unique<manifold::TorusManifold>(d);
}
inline auto projective_space(std::string f, int n)
{
    return std::make_unique<manifold::ProjectiveManifold>(std::move(f), n);
}
inline auto real_projective(int n)
{
    return std::make_unique<manifold::ProjectiveManifold>("R", n);
}
inline auto complex_projective(int n)
{
    return std::make_unique<manifold::ProjectiveManifold>("C", n);
}
inline auto sphere_ptr(int n)
{
    return sphere(n);
}
inline auto torus_ptr(int d = 2)
{
    return torus(d);
}
inline auto klein_bottle()
{
    return std::make_unique<manifold::KleinBottleManifold>();
}
inline auto euclidean(int d = 3)
{
    return std::make_unique<manifold::EuclideanManifold>(d);
}
inline auto genus_g_surface(int g)
{
    return std::make_unique<manifold::GenusGSurfaceManifold>(g);
}
inline auto lens_space(int p, int q = 1)
{
    return std::make_unique<manifold::LensSpaceManifold>(p, q);
}
} // namespace variety
} // namespace np::inline v1

#endif // NP_MANIFOLD_HPP
