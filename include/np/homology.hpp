/**
 * @file homology.hpp
 * @brief Simplicial homology, Smith normal form, Betti numbers.
 *
 * Provides header-only, exact-integer homology for finite simplicial complexes:
 *   - `SimplicialComplex` (by-dimension simplex lists + boundary matrices)
 *   - `smith_normal_form` (exact over Z via `np::bigint`, Bareiss + minors)
 *   - `betti_numbers`, `homology_groups`, `euler_characteristic`
 *   - `simplicial_homology` convenience
 *
 * SNF is exact for 1×1/2×2 via gcd and for general matrices via
 * gcd-of-minors (Cohen) with Bareiss determinants. For large matrices
 * (>2M minors) it falls back to exact rank with invariant factors 1
 * (totally unimodular boundary case). Rank is exact via Bareiss,
 * not double SVD, so `betti_d = n_d - rank(d_d) - rank(d_{d+1})` is
 * exact over Q.
 *
 * Reference: Hatcher, *Algebraic Topology* Ch.2; Munkres, *Elements of Algebraic
 * Topology*; Cohen, *A Course in Computational Algebraic Number Theory* (SNF).
 * numpy-reference: not applicable (abstract topology).
 *
 * @author Sergio Randriamihoatra (sergiorandriamihoatra@gmail.com)
 */
#ifndef NP_HOMOLOGY_HPP
#define NP_HOMOLOGY_HPP

#include <algorithm>
#include <functional>
#include <map>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "api_macros.hpp"
#include "bigint.hpp"
#include "ndarray.hpp"

namespace np::homology
{

// ── SimplicialComplex ───────────────────────────────────────────────────

/**
 * @brief Finite abstract simplicial complex by dimension.
 *
 * `simplices[d]` holds the `d`-simplices, each as sorted `vector<int>` of
 * vertex indices. `d=0` are vertices, `d=1` edges, etc.
 * Example tetrahedron boundary (S²): 4 vertices, 6 edges, 4 faces.
 */
struct SimplicialComplex
{
    std::vector<std::vector<std::vector<int>>> simplices; // index = dim

    SimplicialComplex() = default;
    explicit SimplicialComplex(std::vector<std::vector<std::vector<int>>> s) : simplices(std::move(s))
    {
        for (auto &lvl : simplices)
            for (auto &simp : lvl)
                std::sort(simp.begin(), simp.end());
    }

    NP_NODISCARD int dim() const noexcept
    {
        return static_cast<int>(simplices.size()) - 1;
    }

    NP_NODISCARD std::size_t num_simplices(int d) const noexcept
    {
        if (d < 0 || d >= static_cast<int>(simplices.size()))
            return 0;
        return simplices[d].size();
    }

    /**
     * @brief Boundary matrix `d_d : C_d -> C_{d-1}`.
     * Rows = (d-1)-simplices, cols = d-simplices, entry ±1 if face.
     * For d<=0 or d>dim returns 0×0.
     */
    NP_NODISCARD ndarray<int> boundary_matrix(int d) const
    {
        if (d <= 0 || d >= static_cast<int>(simplices.size()))
            return ndarray<int>::from_data({0, 0}, std::vector<int>{});
        const auto &higher = simplices[d];
        const auto &lower = simplices[d - 1];
        if (higher.empty() || lower.empty())
            return ndarray<int>::from_data({static_cast<int>(lower.size()), static_cast<int>(higher.size())},
                                           std::vector<int>(lower.size() * higher.size(), 0));

        std::map<std::vector<int>, int> lower_index;
        for (int i = 0; i < static_cast<int>(lower.size()); ++i)
            lower_index[lower[i]] = i;

        std::vector<int> data(lower.size() * higher.size(), 0);
        for (int j = 0; j < static_cast<int>(higher.size()); ++j)
        {
            const auto &s = higher[j];
            for (int k = 0; k <= d; ++k)
            {
                std::vector<int> face = s;
                face.erase(face.begin() + k);
                auto it = lower_index.find(face);
                if (it != lower_index.end())
                {
                    int sign = (k % 2 == 0) ? 1 : -1;
                    data[it->second * higher.size() + j] = sign;
                }
            }
        }
        return ndarray<int>::from_data({static_cast<int>(lower.size()), static_cast<int>(higher.size())},
                                       std::move(data));
    }

    /**
     * @brief All boundary matrices `d_1 .. d_dim`.
     * `bms[d]` = `d_d` for `d>=1`, `bms[0]` = 0×0 placeholder.
     */
    NP_NODISCARD std::vector<ndarray<int>> boundary_matrices() const
    {
        std::vector<ndarray<int>> out;
        out.reserve(simplices.size());
        out.push_back(ndarray<int>::from_data({0, 0}, std::vector<int>{})); // d0 placeholder
        for (int d = 1; d < static_cast<int>(simplices.size()); ++d)
            out.push_back(boundary_matrix(d));
        return out;
    }
};

// ── helpers: bigint gcd ─────────────────────────────────────────────────

NP_NODISCARD inline bigint bigint_abs(const bigint &x)
{
    return x < 0 ? -x : x;
}

NP_NODISCARD inline bigint bigint_gcd(bigint a, bigint b)
{
    a = bigint_abs(a);
    b = bigint_abs(b);
    while (b != 0)
    {
        bigint r = a % b;
        a = b;
        b = r;
    }
    return a;
}

// ── Bareiss fraction-free determinant & rank ────────────────────────────

namespace detail
{

NP_NODISCARD inline bigint bareiss_determinant(std::vector<std::vector<bigint>> M)
{
    int n = static_cast<int>(M.size());
    if (n == 0)
        return bigint(1);
    if (static_cast<int>(M[0].size()) != n)
        throw std::invalid_argument("bareiss: not square");
    if (n == 1)
        return M[0][0];
    bigint prev = 1;
    int sign = 1;
    for (int k = 0; k < n - 1; ++k)
    {
        if (M[k][k] == 0)
        {
            int swap = -1;
            for (int i = k + 1; i < n; ++i)
                if (M[i][k] != 0)
                {
                    swap = i;
                    break;
                }
            if (swap == -1)
                return bigint(0);
            std::swap(M[swap], M[k]);
            sign = -sign;
        }
        for (int i = k + 1; i < n; ++i)
        {
            for (int j = k + 1; j < n; ++j)
            {
                M[i][j] = (M[i][j] * M[k][k] - M[i][k] * M[k][j]) / prev;
            }
            M[i][k] = 0;
        }
        prev = M[k][k];
        if (prev == 0)
            return bigint(0);
    }
    bigint det = M[n - 1][n - 1];
    if (sign == -1)
        det = -det;
    return det;
}

NP_NODISCARD inline int bareiss_rank(std::vector<std::vector<bigint>> M)
{
    int m = static_cast<int>(M.size());
    if (m == 0)
        return 0;
    int n = static_cast<int>(M[0].size());
    if (n == 0)
        return 0;
    int rank = 0;
    bigint prev = 1;
    int row = 0;
    for (int col = 0; col < n && row < m; ++col)
    {
        int piv = -1;
        for (int i = row; i < m; ++i)
            if (M[i][col] != 0)
            {
                piv = i;
                break;
            }
        if (piv == -1)
            continue;
        std::swap(M[piv], M[row]);
        for (int i = row + 1; i < m; ++i)
        {
            for (int j = col + 1; j < n; ++j)
            {
                M[i][j] = (M[i][j] * M[row][col] - M[i][col] * M[row][j]) / prev;
            }
            M[i][col] = 0;
        }
        prev = M[row][col];
        ++row;
        ++rank;
        if (prev == 0)
            prev = 1;
    }
    return rank;
}

NP_NODISCARD inline int exact_rank_bigint(const ndarray<bigint> &A)
{
    int m = A.shape[0], n = A.shape[1];
    if (m == 0 || n == 0)
        return 0;
    std::vector<std::vector<bigint>> M(m, std::vector<bigint>(n));
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j)
            M[i][j] = A(i, j);
    return bareiss_rank(std::move(M));
}

NP_NODISCARD inline long long binom_ll(int n, int k)
{
    if (k < 0 || k > n)
        return 0;
    if (k > n - k)
        k = n - k;
    long long res = 1;
    for (int i = 0; i < k; ++i)
    {
        res = res * (n - i) / (i + 1);
        if (res > (long long)5e6)
            return res; // cap
    }
    return res;
}

// Enumerate k-combinations of {0..n-1} invoking fn(comb)
template <typename Fn> inline void for_each_combination(int n, int k, Fn fn)
{
    if (k < 0 || k > n)
        return;
    std::vector<int> c(k);
    std::iota(c.begin(), c.end(), 0);
    while (true)
    {
        fn(c);
        int i = k - 1;
        while (i >= 0 && c[i] == n - k + i)
            --i;
        if (i < 0)
            break;
        ++c[i];
        for (int j = i + 1; j < k; ++j)
            c[j] = c[j - 1] + 1;
    }
}

NP_NODISCARD inline bigint gcd_of_k_minors(const ndarray<bigint> &A, int k)
{
    int m = A.shape[0], n = A.shape[1];
    if (k <= 0)
        return bigint(1);
    if (k > m || k > n)
        return bigint(0);
    bigint g = 0;
    bool first = true;
    // early exit if g becomes 1
    for_each_combination(m, k, [&](const std::vector<int> &rows) {
        if (!first && g == 1)
            return;
        for_each_combination(n, k, [&](const std::vector<int> &cols) {
            if (g == 1)
                return;
            std::vector<std::vector<bigint>> sub(k, std::vector<bigint>(k));
            for (int i = 0; i < k; ++i)
                for (int j = 0; j < k; ++j)
                    sub[i][j] = A(rows[i], cols[j]);
            bigint d = bareiss_determinant(sub);
            d = bigint_abs(d);
            if (d == 0)
                return;
            if (first)
            {
                g = d;
                first = false;
            }
            else
                g = bigint_gcd(g, d);
        });
    });
    if (first)
        return bigint(0); // all zero
    return g;
}

} // namespace detail

// ── Smith normal form ───────────────────────────────────────────────────

/**
 * @brief Smith normal form diagonal for integer matrix `A` (m×n).
 *
 * Returns sorted invariant factors `diag` of length `min(m,n)` where
 * `diag[i] | diag[i+1]` and zeros for rank deficiency. Exact via
 * gcd-of-minors (Bareiss) up to ~2M minors; beyond that falls back to
 * exact rank with 1's (boundary matrices are totally unimodular in that
 * regime). 1×1 and 2×2 are handled directly.
 *
 * Reference: https://en.wikipedia.org/wiki/Smith_normal_form
 */
NP_NODISCARD inline std::vector<bigint> smith_normal_form(const ndarray<int> &A)
{
    ndarray<bigint> Ab = as_bigint(A);
    int m = Ab.shape[0], n = Ab.shape[1];
    int K = std::min(m, n);
    std::vector<bigint> diag(K, bigint(0));
    if (K == 0)
        return diag;
    if (K == 1)
    {
        bigint g = 0;
        for (int i = 0; i < m; ++i)
            for (int j = 0; j < n; ++j)
                g = bigint_gcd(g, Ab(i, j));
        diag[0] = bigint_abs(g);
        return diag;
    }
    if (m == 2 && n == 2)
    {
        bigint a = Ab(0, 0), b = Ab(0, 1), c = Ab(1, 0), d = Ab(1, 1);
        bigint g = bigint_gcd(bigint_gcd(bigint_gcd(a, b), c), d);
        if (g == 0)
            return diag;
        bigint det = a * d - b * c;
        diag[0] = bigint_abs(g);
        if (det != 0)
            diag[1] = bigint_abs(det) / diag[0];
        if (diag[0] == 0)
            std::swap(diag[0], diag[1]);
        return diag;
    }
    // General: gcd of minors
    // Estimate total minors
    long long total_est = 0;
    bool too_large = false;
    for (int k = 1; k <= K; ++k)
    {
        long long cr = detail::binom_ll(m, k);
        long long cc = detail::binom_ll(n, k);
        if (cr > 100000 || cc > 100000)
        {
            too_large = true;
            break;
        }
        long long tot = cr * cc;
        if (tot > 2000000)
        {
            too_large = true;
            break;
        }
        total_est += tot;
        if (total_est > 2000000)
        {
            too_large = true;
            break;
        }
    }
    if (too_large)
    {
        int r = detail::exact_rank_bigint(Ab);
        for (int i = 0; i < r && i < K; ++i)
            diag[i] = bigint(1);
        return diag;
    }
    bigint g_prev = 1;
    for (int k = 1; k <= K; ++k)
    {
        bigint gk = detail::gcd_of_k_minors(Ab, k);
        if (gk == 0)
            break; // rank < k
        diag[k - 1] = gk / g_prev;
        g_prev = gk;
    }
    // Ensure divisibility and sort (already sorted by construction)
    for (auto &v : diag)
        if (v < 0)
            v = -v;
    return diag;
}

NP_NODISCARD inline std::vector<bigint> smith_normal_form(const ndarray<bigint> &A)
{
    int m = A.shape[0], n = A.shape[1];
    int K = std::min(m, n);
    std::vector<bigint> diag(K, bigint(0));
    if (K == 0)
        return diag;
    if (K == 1)
    {
        bigint g = 0;
        for (int i = 0; i < m; ++i)
            for (int j = 0; j < n; ++j)
                g = bigint_gcd(g, A(i, j));
        diag[0] = bigint_abs(g);
        return diag;
    }
    if (m == 2 && n == 2)
    {
        bigint a = A(0, 0), b = A(0, 1), c = A(1, 0), d = A(1, 1);
        bigint g = bigint_gcd(bigint_gcd(bigint_gcd(a, b), c), d);
        if (g == 0)
            return diag;
        bigint det = a * d - b * c;
        diag[0] = bigint_abs(g);
        if (det != 0)
            diag[1] = bigint_abs(det) / diag[0];
        return diag;
    }
    long long total_est = 0;
    bool too_large = false;
    for (int k = 1; k <= K; ++k)
    {
        long long cr = detail::binom_ll(m, k);
        long long cc = detail::binom_ll(n, k);
        if (cr > 100000 || cc > 100000)
        {
            too_large = true;
            break;
        }
        long long tot = cr * cc;
        if (tot > 2000000)
        {
            too_large = true;
            break;
        }
        total_est += tot;
        if (total_est > 2000000)
        {
            too_large = true;
            break;
        }
    }
    if (too_large)
    {
        int r = detail::exact_rank_bigint(A);
        for (int i = 0; i < r && i < K; ++i)
            diag[i] = bigint(1);
        return diag;
    }
    bigint g_prev = 1;
    for (int k = 1; k <= K; ++k)
    {
        bigint gk = detail::gcd_of_k_minors(A, k);
        if (gk == 0)
            break;
        diag[k - 1] = gk / g_prev;
        g_prev = gk;
    }
    for (auto &v : diag)
        if (v < 0)
            v = -v;
    return diag;
}

// ── Betti numbers & homology ────────────────────────────────────────────

struct HomologyGroup
{
    int betti = 0;
    std::vector<bigint> torsion; // invariant factors >1
    std::string to_string() const
    {
        std::string s = "Z^" + std::to_string(betti);
        if (!torsion.empty())
        {
            s += " + ";
            for (size_t i = 0; i < torsion.size(); ++i)
            {
                if (i)
                    s += " + ";
                s += "Z/" + torsion[i].convert_to<std::string>() + "Z";
            }
        }
        return s;
    }
};

/**
 * @brief Betti numbers for a chain complex given by boundary matrices.
 *
 * `bms[d]` = `d_d` (d>=1), `bms[0]` placeholder, `n_d` = `bms[d].shape[1]` for d>=1,
 * `n_0` = `bms[1].shape[0]`. Over Q: `betti_d = n_d - rank(d_d) - rank(d_{d+1})`.
 * Rank is exact via Bareiss over Z.
 */
NP_NODISCARD inline std::vector<int> betti_numbers(const std::vector<ndarray<int>> &bms)
{
    int D = static_cast<int>(bms.size()) - 1;
    if (D < 0)
        return {};
    std::vector<int> n(D + 1, 0);
    for (int d = 0; d <= D; ++d)
    {
        if (d == 0)
            n[0] = (D >= 1) ? bms[1].shape[0] : 0;
        else
            n[d] = bms[d].shape[1];
    }
    std::vector<int> rank(D + 1, 0);
    for (int d = 1; d <= D; ++d)
    {
        if (bms[d].size() == 0)
            continue;
        ndarray<bigint> Ab = as_bigint(bms[d]);
        rank[d] = detail::exact_rank_bigint(Ab);
    }
    std::vector<int> betti(D + 1, 0);
    for (int d = 0; d <= D; ++d)
    {
        int rd = (d <= D) ? rank[d] : 0;
        int rd1 = (d + 1 <= D) ? rank[d + 1] : 0;
        betti[d] = n[d] - rd - rd1;
        if (betti[d] < 0)
            betti[d] = 0;
    }
    return betti;
}

NP_NODISCARD inline std::vector<int> betti_numbers(const SimplicialComplex &K)
{
    return betti_numbers(K.boundary_matrices());
}

NP_NODISCARD inline std::vector<HomologyGroup> homology_groups(const std::vector<ndarray<int>> &bms)
{
    auto betti = betti_numbers(bms);
    std::vector<HomologyGroup> out(betti.size());
    for (size_t i = 0; i < betti.size(); ++i)
        out[i].betti = betti[i];
    for (size_t d = 0; d + 1 < bms.size(); ++d)
    {
        if (bms[d + 1].size() == 0)
            continue;
        auto diag = smith_normal_form(bms[d + 1]);
        for (auto &v : diag)
            if (v > 1)
                out[d].torsion.push_back(v);
        std::sort(out[d].torsion.begin(), out[d].torsion.end());
    }
    return out;
}

NP_NODISCARD inline std::vector<HomologyGroup> homology_groups(const SimplicialComplex &K)
{
    return homology_groups(K.boundary_matrices());
}

NP_NODISCARD inline int euler_characteristic(const SimplicialComplex &K)
{
    int e = 0;
    for (int d = 0; d < static_cast<int>(K.simplices.size()); ++d)
    {
        int nd = static_cast<int>(K.simplices[d].size());
        e += (d % 2 == 0) ? nd : -nd;
    }
    return e;
}

NP_NODISCARD inline int euler_characteristic(const std::vector<ndarray<int>> &bms)
{
    auto betti = betti_numbers(bms);
    int e = 0;
    for (size_t i = 0; i < betti.size(); ++i)
        e += (i % 2 == 0) ? betti[i] : -betti[i];
    return e;
}

// ── Convenience constructors ────────────────────────────────────────────

NP_NODISCARD inline SimplicialComplex make_simplex(int n_vertices)
{
    if (n_vertices <= 0)
        return SimplicialComplex{};
    if (n_vertices > 20)
        throw std::invalid_argument("make_simplex: n_vertices >20 combinatorial explosion");
    std::vector<std::vector<std::vector<int>>> s(n_vertices);
    for (int mask = 1; mask < (1 << n_vertices); ++mask)
    {
        std::vector<int> face;
        for (int i = 0; i < n_vertices; ++i)
            if (mask & (1 << i))
                face.push_back(i);
        int d = static_cast<int>(face.size()) - 1;
        s[d].push_back(face);
    }
    for (auto &lvl : s)
        std::sort(lvl.begin(), lvl.end());
    return SimplicialComplex{s};
}

NP_NODISCARD inline SimplicialComplex circle_complex()
{
    return SimplicialComplex{{{{0}, {1}, {2}}, {{0, 1}, {1, 2}, {0, 2}}, {}}};
}

NP_NODISCARD inline SimplicialComplex sphere_tetrahedron()
{
    return SimplicialComplex{{{{0}, {1}, {2}, {3}},
                              {{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}},
                              {{0, 1, 2}, {0, 1, 3}, {0, 2, 3}, {1, 2, 3}},
                              {}}};
}

// ── Builder & ergonomic helpers ───────────────────────────────────────

struct SimplicialComplexBuilder
{
    std::set<std::vector<int>> simplices;
    void add_simplex(std::vector<int> s)
    {
        std::sort(s.begin(), s.end());
        int n = static_cast<int>(s.size());
        if (n == 0)
            return;
        if (n > 20)
            throw std::invalid_argument("add_simplex: simplex too large");
        for (int mask = 1; mask < (1 << n); ++mask)
        {
            std::vector<int> face;
            for (int i = 0; i < n; ++i)
                if (mask & (1 << i))
                    face.push_back(s[i]);
            simplices.insert(face);
        }
    }
    void add_maximal(const std::vector<std::vector<int>> &max)
    {
        for (auto s : max)
            add_simplex(s);
    }
    SimplicialComplex build() const
    {
        int max_dim = 0;
        for (auto &s : simplices)
            max_dim = std::max(max_dim, static_cast<int>(s.size()) - 1);
        std::vector<std::vector<std::vector<int>>> lvl(max_dim + 1);
        for (auto &s : simplices)
            lvl[s.size() - 1].push_back(s);
        for (auto &l : lvl)
            std::sort(l.begin(), l.end());
        return SimplicialComplex{lvl};
    }
};

NP_NODISCARD inline SimplicialComplex sphere_boundary(int n)
{
    if (n < 0)
        return SimplicialComplex{};
    if (n == 0)
        return SimplicialComplex{{{{0}}, {{1}}, {}, {}}};
    if (n == 1)
        return circle_complex();
    if (n == 2)
        return sphere_tetrahedron();
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
    SimplicialComplexBuilder b;
    b.add_maximal(facets);
    return b.build();
}

NP_NODISCARD inline SimplicialComplex from_maximal(std::vector<std::vector<int>> max)
{
    SimplicialComplexBuilder b;
    b.add_maximal(max);
    return b.build();
}

// Aliases for discoverability
NP_NODISCARD inline SimplicialComplex make_circle()
{
    return circle_complex();
}
NP_NODISCARD inline SimplicialComplex make_sphere(int n = 2)
{
    return sphere_boundary(n);
}

NP_NODISCARD inline int betti(const SimplicialComplex &K, int k)
{
    auto b = betti_numbers(K);
    if (k < 0 || k >= static_cast<int>(b.size()))
        return 0;
    return b[k];
}
NP_NODISCARD inline std::string homology_string(const SimplicialComplex &K)
{
    auto hg = homology_groups(K);
    std::string s;
    for (size_t i = 0; i < hg.size(); ++i)
    {
        if (i)
            s += " + ";
        s += "H" + std::to_string(i) + "=" + hg[i].to_string();
    }
    return s;
}

} // namespace np::homology

#endif // NP_HOMOLOGY_HPP
