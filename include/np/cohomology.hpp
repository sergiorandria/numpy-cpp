/**
 * @file cohomology.hpp
 * @brief Cohomology ring, cup product, Poincaré duality, Künneth and UCT.
 *
 * Extends `np::homology` with dual cohomology:
 *   - `CohomologyGroup` (`betti`, `torsion`) via Universal Coefficients:
 *     `H^n ≅ Hom(H_n,Z) ⊕ Ext(H_{n-1},Z)` – `betti^n = betti_n`,
 *     `torsion^n = torsion_{n-1}`.
 *   - `cohomology_groups`, `betti_cohomology`, `euler via cohomology`
 *   - `CohomologyRing` – cup table computed at cochain level over Q
 *     (Alexander–Whitney), presentations for classical spaces
 *     (S^n, T^n, CP^n); `inconclusive=true` with torsion, oversized
 *     complexes, or multi-term products the int table cannot express.
 *     Boundary-matrix-only input (`bms`) cannot support cup products
 *     (front/back face incidence needs vertex labels) and stays inconclusive.
 *   - `cup_product(K,p,q, a,b)` → class index in `H^{p+q}`
 *   - `poincare_pairing`, `intersection_form` (closed oriented 2n-manifolds)
 *   - `kunneth_cohomology` and `universal_coefficients` helpers
 *   - `cohomology_ring_string`
 *
 * The cup product for arbitrary simplicial complexes is computed at
 * cochain level (Alexander–Whitney) over Q; presentations for the
 * classical manifolds are kept as ring descriptions.
 *
 * Reference: Hatcher Ch.3, Bott–Tu, May *Concise*.
 *
 * @author Sergio Randriamihoatra (sergiorandriamihoatra@gmail.com)
 */
#ifndef NP_COHOMOLOGY_HPP
#define NP_COHOMOLOGY_HPP

#include <algorithm>
#include <limits>
#include <map>
#include <string>
#include <vector>

#include "api_macros.hpp"
#include "bigint.hpp"
#include "homology.hpp"

// Cup-product tuning (macros, no magic numbers in logic).
// Exact rational elimination above this many simplices degrades to an
// inconclusive ring instead of risking coefficient blowup.
#define NP_COHOMOLOGY_CUP_MAX_SIMPLEX 4096

namespace np::cohomology
{

struct CohomologyGroup
{
    int betti = 0;
    std::vector<bigint> torsion;
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
 * @brief Cohomology via UCT: H^n = Hom(H_n,Z) ⊕ Ext(H_{n-1},Z).
 */
NP_NODISCARD inline std::vector<CohomologyGroup> cohomology_groups(const std::vector<np::ndarray<int>> &bms)
{
    auto hg = homology::homology_groups(bms);
    std::vector<CohomologyGroup> out(hg.size());
    for (size_t n = 0; n < hg.size(); ++n)
    {
        out[n].betti = hg[n].betti;
        if (n > 0)
            out[n].torsion = hg[n - 1].torsion;
    }
    return out;
}

NP_NODISCARD inline std::vector<CohomologyGroup> cohomology_groups(const homology::SimplicialComplex &K)
{
    return cohomology_groups(K.boundary_matrices());
}

NP_NODISCARD inline std::vector<int> betti_cohomology(const homology::SimplicialComplex &K)
{
    auto cg = cohomology_groups(K);
    std::vector<int> b(cg.size());
    for (size_t i = 0; i < cg.size(); ++i)
        b[i] = cg[i].betti;
    return b;
}

struct CohomologyRing
{
    std::vector<CohomologyGroup> groups;
    // cup[p][q][a][b] = c index in H^{p+q} or -1 if zero; -2 inconclusive
    std::vector<std::vector<std::vector<std::vector<int>>>> cup;
    bool inconclusive = false;
    std::string presentation; // e.g. "Z[h]/(h^{3})" for CP2
    std::string to_string() const
    {
        std::string s;
        for (size_t i = 0; i < groups.size(); ++i)
        {
            if (i)
                s += " | ";
            s += "H^" + std::to_string(i) + "=" + groups[i].to_string();
        }
        if (!presentation.empty())
            s += "  ring: " + presentation;
        if (inconclusive)
            s += " (cup inconclusive)";
        return s;
    }
};

namespace detail
{

NP_NODISCARD inline int effective_dim(const std::vector<homology::HomologyGroup> &hg)
{
    int D = static_cast<int>(hg.size()) - 1;
    while (D > 0 && hg[D].betti == 0 && hg[D].torsion.empty())
        --D;
    return D;
}

NP_NODISCARD inline bool is_torus_pattern(const std::vector<homology::HomologyGroup> &hg)
{
    int D = effective_dim(hg);
    for (int k = 0; k <= D; ++k)
    {
        // binomial(D,k)
        long long num = 1, den = 1;
        for (int i = 0; i < k; ++i)
        {
            num *= (D - i);
            den *= (k - i);
        }
        int bin = (k == 0) ? 1 : static_cast<int>(num / den);
        if (k >= static_cast<int>(hg.size()) || hg[k].betti != bin)
            return false;
        if (!hg[k].torsion.empty())
            return false;
    }
    // trailing beyond D must be zero
    for (int k = D + 1; k < static_cast<int>(hg.size()); ++k)
        if (hg[k].betti != 0 || !hg[k].torsion.empty())
            return false;
    return D >= 0;
}

NP_NODISCARD inline bool is_sphere_pattern(const std::vector<homology::HomologyGroup> &hg)
{
    int D = effective_dim(hg);
    if (D < 0)
        return false;
    for (int k = 0; k <= D; ++k)
    {
        int bet = (k == 0 || k == D) ? 1 : 0;
        if (k >= static_cast<int>(hg.size()) || hg[k].betti != bet)
            return false;
        if (!hg[k].torsion.empty())
            return false;
    }
    for (int k = D + 1; k < static_cast<int>(hg.size()); ++k)
        if (hg[k].betti != 0 || !hg[k].torsion.empty())
            return false;
    return true;
}

NP_NODISCARD inline bool is_cp_pattern(const std::vector<homology::HomologyGroup> &hg, int &n_out)
{
    int D = effective_dim(hg);
    if (D % 2 == 1)
        return false;
    int n = D / 2;
    for (int k = 0; k <= D; ++k)
    {
        if (k % 2 == 1 && hg[k].betti != 0)
            return false;
        if (k % 2 == 0 && hg[k].betti != 1)
            return false;
        if (!hg[k].torsion.empty())
            return false;
    }
    for (int k = D + 1; k < static_cast<int>(hg.size()); ++k)
        if (hg[k].betti != 0 || !hg[k].torsion.empty())
            return false;
    n_out = n;
    return true;
}

} // namespace detail

namespace detail
{

// ── Exact rationals (bigint numerator/denominator) for cochain algebra ──

struct Frac
{
    bigint num{0};
    bigint den{1};

    Frac() = default;
    Frac(int n) : num(n), den(1)
    {
    }
    Frac(const bigint &n) : num(n), den(1)
    {
    }
    Frac(const bigint &n, const bigint &d)
    {
        assign(n, d);
    }
    void assign(const bigint &n, const bigint &d)
    {
        if (d == 0)
        {
            throw std::invalid_argument("Frac: zero denominator");
        }
        bigint nn = n, dd = d;
        if (dd < 0)
        {
            nn = -nn;
            dd = -dd;
        }
        if (nn == 0)
        {
            num = 0;
            den = 1;
            return;
        }
        const bigint g = homology::bigint_gcd(nn, dd);
        num = nn / g;
        den = dd / g;
    }
    NP_NODISCARD bool is_zero() const noexcept
    {
        return num == 0;
    }
};

NP_NODISCARD inline Frac operator+(const Frac &a, const Frac &b)
{
    return Frac(a.num * b.den + b.num * a.den, a.den * b.den);
}
NP_NODISCARD inline Frac operator-(const Frac &a, const Frac &b)
{
    return Frac(a.num * b.den - b.num * a.den, a.den * b.den);
}
NP_NODISCARD inline Frac operator-(const Frac &a)
{
    return Frac(-a.num, a.den);
}
NP_NODISCARD inline Frac operator*(const Frac &a, const Frac &b)
{
    return Frac(a.num * b.num, a.den * b.den);
}
NP_NODISCARD inline Frac operator/(const Frac &a, const Frac &b)
{
    if (b.num == 0)
    {
        throw std::invalid_argument("Frac: division by zero");
    }
    return Frac(a.num * b.den, a.den * b.num);
}

/// RREF in place over Q; returns pivot columns. Exact (no rounding).
inline std::vector<int> rref(std::vector<std::vector<Frac>> &m)
{
    const int rows = static_cast<int>(m.size());
    if (rows == 0)
    {
        return {};
    }
    const int cols = static_cast<int>(m[0].size());
    std::vector<int> pivots;
    int r = 0;
    for (int c = 0; c < cols && r < rows; ++c)
    {
        int piv = -1;
        for (int i = r; i < rows; ++i)
        {
            if (!m[i][c].is_zero())
            {
                piv = i;
                break;
            }
        }
        if (piv < 0)
        {
            continue;
        }
        std::swap(m[r], m[piv]);
        const Frac inv(m[r][c].den, m[r][c].num); // 1/pivot; pivot is nonzero
        for (int j = c; j < cols; ++j)
        {
            m[r][j] = m[r][j] * inv;
        }
        for (int i = 0; i < rows; ++i)
        {
            if (i != r && !m[i][c].is_zero())
            {
                const Frac f = m[i][c];
                for (int j = c; j < cols; ++j)
                {
                    m[i][j] = m[i][j] - f * m[r][j];
                }
            }
        }
        pivots.push_back(c);
        ++r;
    }
    return pivots;
}

/// Basis of ker(rows) over Q, one vector per free variable.
/// ncols is the ambient dimension (needed when rows is empty: no
/// constraints means the whole space, not the zero space).
NP_NODISCARD inline std::vector<std::vector<Frac>> nullspace(std::vector<std::vector<Frac>> rows, int ncols)
{
    const int n = ncols;
    if (n <= 0)
    {
        return {};
    }
    // Drop all-zero rows (no constraints).
    std::vector<std::vector<Frac>> m;
    for (auto &row : rows)
    {
        bool any = false;
        for (auto &x : row)
        {
            if (!x.is_zero())
            {
                any = true;
                break;
            }
        }
        if (any)
        {
            m.push_back(row);
        }
    }
    if (m.empty())
    {
        // Whole space: standard basis.
        std::vector<std::vector<Frac>> out(n, std::vector<Frac>(n, Frac(0)));
        for (int i = 0; i < n; ++i)
        {
            out[i][i] = Frac(1);
        }
        return out;
    }
    const auto pivots = rref(m);
    std::vector<bool> is_pivot(n, false);
    for (int p : pivots)
    {
        is_pivot[p] = true;
    }
    // Row index per pivot column.
    std::vector<int> pivot_row(n, -1);
    for (int i = 0; i < static_cast<int>(m.size()); ++i)
    {
        for (int c = 0; c < n; ++c)
        {
            if (!m[i][c].is_zero())
            {
                pivot_row[c] = i;
                break;
            }
        }
    }
    std::vector<std::vector<Frac>> out;
    for (int f = 0; f < n; ++f)
    {
        if (is_pivot[f])
        {
            continue;
        }
        std::vector<Frac> v(n, Frac(0));
        v[f] = Frac(1);
        for (int p = 0; p < n; ++p)
        {
            if (is_pivot[p])
            {
                v[p] = -m[pivot_row[p]][f];
            }
        }
        out.push_back(std::move(v));
    }
    return out;
}

/// Basis of the column space of A (m×n): original columns at pivot positions.
NP_NODISCARD inline std::vector<std::vector<Frac>> column_basis(const std::vector<std::vector<Frac>> &A)
{
    if (A.empty() || A[0].empty())
    {
        return {};
    }
    auto m = A;
    const auto pivots = rref(m);
    const int rows = static_cast<int>(A.size());
    std::vector<std::vector<Frac>> out;
    for (int p : pivots)
    {
        std::vector<Frac> col(rows);
        for (int i = 0; i < rows; ++i)
        {
            col[i] = A[i][p];
        }
        out.push_back(std::move(col));
    }
    return out;
}

NP_NODISCARD inline std::size_t row_rank(std::vector<std::vector<Frac>> rows)
{
    if (rows.empty() || rows[0].empty())
    {
        return 0;
    }
    return rref(rows).size();
}

/**
 * @brief Solve E·c = z over Q where E's columns are basis vectors.
 * @return Coordinates c; throws std::logic_error if inconsistent
 *         (cannot happen for cocycles of a genuine complex).
 */
NP_NODISCARD inline std::vector<Frac> solve_columns(const std::vector<std::vector<Frac>> &cols,
                                                    const std::vector<Frac> &z)
{
    const std::size_t n = z.size();
    const std::size_t t = cols.size();
    std::vector<std::vector<Frac>> m(n, std::vector<Frac>(t + 1, Frac(0)));
    for (std::size_t i = 0; i < n; ++i)
    {
        for (std::size_t j = 0; j < t; ++j)
        {
            m[i][j] = cols[j][i];
        }
        m[i][t] = z[i];
    }
    rref(m);
    std::vector<Frac> c(t, Frac(0));
    for (std::size_t i = 0; i < n; ++i)
    {
        int piv = -1;
        for (std::size_t j = 0; j < t; ++j)
        {
            if (!m[i][j].is_zero())
            {
                piv = static_cast<int>(j);
                break;
            }
        }
        if (piv < 0)
        {
            if (!m[i][t].is_zero())
            {
                throw std::logic_error("solve_columns: inconsistent system (not a cocycle?)");
            }
            continue;
        }
        c[piv] = m[i][t];
    }
    return c;
}

// ── Rational cohomology with Alexander–Whitney cup product ──

/// Cup data over Q: H-basis per degree + cup coordinates in the target basis.
struct RationalCohomology
{
    int D = -1;
    std::vector<int> betti;                                                    ///< Rational Betti per degree.
    std::vector<std::vector<std::vector<Frac>>> hbasis;                        ///< hbasis[p][i]: i-th H^p rep.
    std::vector<std::vector<std::vector<Frac>>> bbasis;                        ///< bbasis[p][j]: coboundary basis.
    std::vector<std::vector<std::vector<std::vector<std::vector<Frac>>>>> cup; ///< cup[p][q][a][b]: coords.
    bool ok = false;
};

/**
 * @brief Cohomology over Q with cochain-level Alexander–Whitney cup product.
 *
 * Coboundary δ^p is the transpose of ∂_{p+1}; H^p = ker δ^p / im δ^{p-1}.
 * (α⌣β)[v_0..v_{p+q}] = α[v_0..v_p]·β[v_p..v_{p+q}].
 * Throws std::invalid_argument if the complex is not closed under faces.
 */
NP_NODISCARD inline RationalCohomology rational_cohomology(const homology::SimplicialComplex &K)
{
    RationalCohomology rc;
    const int D = K.dim();
    if (D < 0)
    {
        return rc;
    }
    rc.D = D;
    std::vector<int> n(D + 1, 0);
    std::vector<std::map<std::vector<int>, int>> index(D + 1);
    for (int d = 0; d <= D; ++d)
    {
        n[d] = static_cast<int>(K.simplices[d].size());
        for (int i = 0; i < n[d]; ++i)
        {
            index[d][K.simplices[d][i]] = i;
        }
    }
    // Coboundary matrices: delta^p (n_{p+1} rows × n_p cols) = transpose of d_{p+1}.
    std::vector<std::vector<std::vector<Frac>>> delta(D + 1);
    for (int p = 0; p <= D; ++p)
    {
        if (p + 1 > D || n[p] == 0 || n[p + 1] == 0)
        {
            delta[p].clear();
            continue;
        }
        const auto bm = K.boundary_matrix(p + 1); // n_p rows × n_{p+1} cols
        delta[p].assign(n[p + 1], std::vector<Frac>(n[p], Frac(0)));
        for (int i = 0; i < n[p]; ++i)
        {
            for (int j = 0; j < n[p + 1]; ++j)
            {
                const int v = bm(i, j);
                if (v != 0)
                {
                    delta[p][j][i] = Frac(v);
                }
            }
        }
    }
    rc.betti.assign(D + 1, 0);
    rc.hbasis.assign(D + 1, {});
    rc.bbasis.assign(D + 1, {});
    for (int p = 0; p <= D; ++p)
    {
        if (n[p] == 0)
        {
            continue;
        }
        const auto ker = nullspace(delta[p], n[p]); // cocycle representatives
        std::vector<std::vector<Frac>> img;
        if (p > 0 && !delta[p - 1].empty())
        {
            img = column_basis(delta[p - 1]); // coboundaries (⊆ ker since δ²=0)
        }
        rc.bbasis[p] = img;
        // Greedy complement: keep kernel vectors that add rank over img+kept.
        std::vector<std::vector<Frac>> kept;
        for (const auto &kv : ker)
        {
            auto trial = img;
            trial.insert(trial.end(), kept.begin(), kept.end());
            const std::size_t before = row_rank(trial);
            trial.push_back(kv);
            if (row_rank(trial) > before)
            {
                kept.push_back(kv);
            }
        }
        rc.hbasis[p] = kept;
        rc.betti[p] = static_cast<int>(kept.size());
    }
    // Alexander–Whitney cup products, reduced in the H^{p+q} basis.
    rc.cup.assign(D + 1, {});
    for (int p = 0; p <= D; ++p)
    {
        rc.cup[p].assign(D + 1, {});
        for (int q = 0; q <= D; ++q)
        {
            const int r = p + q;
            if (r > D || rc.hbasis[p].empty() || rc.hbasis[q].empty() || rc.betti[r] == 0)
            {
                continue;
            }
            const int bp = static_cast<int>(rc.hbasis[p].size());
            const int bq = static_cast<int>(rc.hbasis[q].size());
            rc.cup[p][q].assign(bp, {});
            // Reduction basis: coboundaries + H representatives in degree r.
            std::vector<std::vector<Frac>> red = rc.bbasis[r];
            red.insert(red.end(), rc.hbasis[r].begin(), rc.hbasis[r].end());
            const std::size_t nb = rc.bbasis[r].size();
            for (int a = 0; a < bp; ++a)
            {
                rc.cup[p][q][a].assign(bq, {});
                for (int b = 0; b < bq; ++b)
                {
                    std::vector<Frac> z(n[r], Frac(0));
                    for (int s = 0; s < n[r]; ++s)
                    {
                        const auto &sig = K.simplices[r][s];
                        std::vector<int> front(sig.begin(), sig.begin() + p + 1);
                        std::vector<int> back(sig.begin() + p, sig.end());
                        const auto itf = index[p].find(front);
                        const auto itb = index[q].find(back);
                        if (itf == index[p].end() || itb == index[q].end())
                        {
                            throw std::invalid_argument("rational_cohomology: complex not closed under faces");
                        }
                        z[s] = rc.hbasis[p][a][itf->second] * rc.hbasis[q][b][itb->second];
                    }
                    const auto coords = solve_columns(red, z);
                    rc.cup[p][q][a][b].assign(coords.begin() + nb, coords.end());
                }
            }
        }
    }
    rc.ok = true;
    return rc;
}

} // namespace detail

NP_NODISCARD inline CohomologyRing cohomology_ring(const homology::SimplicialComplex &K)
{
    auto hg = homology::homology_groups(K);
    auto cg_vec = cohomology_groups(K);
    CohomologyRing R;
    R.groups = cg_vec;
    const int D = static_cast<int>(cg_vec.size()) - 1;
    R.cup.assign(D + 1, std::vector<std::vector<std::vector<int>>>(D + 1));
    // Torsion defeats integral reading of the table (e.g. RP² has a²≠0 mod 2
    // while the rational table is zero): flag inconclusive, table stays rational.
    for (const auto &g : hg)
    {
        if (!g.torsion.empty())
        {
            R.inconclusive = true;
            break;
        }
    }
    // Simplex-count guard: exact rational elimination is exponential in the
    // worst case; above the cap degrade to an inconclusive ring, not garbage.
    std::size_t total = 0;
    for (int d = 0; d <= K.dim(); ++d)
    {
        total += K.num_simplices(d);
    }
    if (total <= static_cast<std::size_t>(NP_COHOMOLOGY_CUP_MAX_SIMPLEX))
    {
        const auto rc = detail::rational_cohomology(K);
        for (int p = 0; p <= D; ++p)
        {
            for (int q = 0; q <= D; ++q)
            {
                const int r = p + q;
                if (r < 0 || r > D)
                {
                    continue;
                }
                const int bp = cg_vec[p].betti;
                const int bq = cg_vec[q].betti;
                if (bp == 0 || bq == 0)
                {
                    continue;
                }
                R.cup[p][q].assign(bp, std::vector<int>(bq, -1));
                if (r >= static_cast<int>(rc.cup.size()) || p >= static_cast<int>(rc.cup.size()) ||
                    q >= static_cast<int>(rc.cup[p].size()))
                {
                    continue;
                }
                const auto &tab = rc.cup[p][q];
                for (int a = 0; a < bp && a < static_cast<int>(tab.size()); ++a)
                {
                    for (int b = 0; b < bq && b < static_cast<int>(tab[a].size()); ++b)
                    {
                        int first = -1;
                        bool multi = false;
                        for (int c = 0; c < static_cast<int>(tab[a][b].size()); ++c)
                        {
                            if (!tab[a][b][c].is_zero())
                            {
                                if (first < 0)
                                {
                                    first = c;
                                }
                                else
                                {
                                    multi = true;
                                    break;
                                }
                            }
                        }
                        if (multi)
                        {
                            // Genuine combination: the int table cannot express it.
                            R.cup[p][q][a][b] = -2;
                            R.inconclusive = true;
                        }
                        else
                        {
                            R.cup[p][q][a][b] = first; // -1 when zero
                        }
                    }
                }
            }
        }
    }
    else
    {
        R.inconclusive = true;
    }
    // Presentations for the classical patterns (still true as ring descriptions).
    if (detail::is_torus_pattern(hg))
    {
        R.presentation = "Λ[a,b] (exterior)";
    }
    else if (detail::is_sphere_pattern(hg))
    {
        R.presentation = "Z[x]/(x^2) |x|=" + std::to_string(detail::effective_dim(hg));
    }
    else
    {
        int ncp = 0;
        if (detail::is_cp_pattern(hg, ncp))
        {
            R.presentation = "Z[h]/(h^" + std::to_string(ncp + 1) + ") |h|=2";
        }
    }
    return R;
}

/**
 * @brief Rank of the cup-product pairing H^p × H^q → H^{p+q} over Q.
 *
 * Basis-independent (unlike raw table indices): the rank of the set of
 * coordinate vectors {a⌣b}. Returns 0 when any side vanishes, -1 when
 * uncomputable (oversized complex).
 */
NP_NODISCARD inline int cup_pairing_rank(const homology::SimplicialComplex &K, int p, int q)
{
    const int D = K.dim();
    if (p < 0 || q < 0 || p > D || q > D || p + q > D)
    {
        return 0;
    }
    std::size_t total = 0;
    for (int d = 0; d <= D; ++d)
    {
        total += K.num_simplices(d);
    }
    if (total > static_cast<std::size_t>(NP_COHOMOLOGY_CUP_MAX_SIMPLEX))
    {
        return -1;
    }
    const auto rc = detail::rational_cohomology(K);
    if (p >= static_cast<int>(rc.cup.size()) || q >= static_cast<int>(rc.cup[p].size()))
    {
        return 0;
    }
    std::vector<std::vector<detail::Frac>> vecs;
    for (const auto &row : rc.cup[p][q])
    {
        for (const auto &coords : row)
        {
            vecs.push_back(coords);
        }
    }
    return static_cast<int>(detail::row_rank(std::move(vecs)));
}

NP_NODISCARD inline CohomologyRing cohomology_ring(const std::vector<ndarray<int>> &bms)
{
    // Boundary matrices alone cannot support cup products: the
    // Alexander–Whitney formula needs front/back face incidence, i.e.
    // vertex labels. This overload stays inconclusive by design.
    auto hg = homology::homology_groups(bms);
    CohomologyRing R;
    std::vector<CohomologyGroup> cg(hg.size());
    for (size_t n = 0; n < hg.size(); ++n)
    {
        cg[n].betti = hg[n].betti;
        if (n > 0)
            cg[n].torsion = hg[n - 1].torsion;
    }
    R.groups = cg;
    int D = static_cast<int>(cg.size()) - 1;
    R.cup.assign(D + 1, std::vector<std::vector<std::vector<int>>>(D + 1));
    for (int p = 0; p <= D; ++p)
        for (int q = 0; q <= D; ++q)
        {
            int r = p + q;
            if (r > D || r < 0)
                continue;
            int bp = cg[p].betti, bq = cg[q].betti, br = cg[r].betti;
            if (bp == 0 || bq == 0 || br == 0)
                continue;
            R.cup[p][q].assign(bp, std::vector<int>(bq, -1));
            if (p == 0 || q == 0)
                for (int a = 0; a < bp; ++a)
                    for (int b = 0; b < bq; ++b)
                        R.cup[p][q][a][b] = 0;
        }
    R.inconclusive = true;
    return R;
}

/**
 * @brief Cup product `a∈H^p, b∈H^q → c∈H^{p+q}` index, or -1 if zero, -2 inconclusive.
 */
NP_NODISCARD inline int cup_product(const homology::SimplicialComplex &K, int p, int q, int a, int b)
{
    auto R = cohomology_ring(K);
    int D = static_cast<int>(R.groups.size()) - 1;
    if (p < 0 || q < 0 || p > D || q > D)
        return -2;
    int r = p + q;
    if (r > D)
        return -1;
    if (p >= static_cast<int>(R.cup.size()) || q >= static_cast<int>(R.cup[p].size()))
        return -2;
    if (R.cup[p][q].empty())
        return -2;
    if (a < 0 || a >= static_cast<int>(R.cup[p][q].size()))
        return -2;
    if (b < 0 || b >= static_cast<int>(R.cup[p][q][a].size()))
        return -2;
    int v = R.cup[p][q][a][b];
    if (R.inconclusive && v == -1)
        return -2;
    return v;
}

namespace detail
{
// Cup-pairing block M_{ab} = <a⌣b,[M]> for fixed (p,q) with p+q=n, read off
// rational_cohomology()'s Alexander–Whitney table. Returns empty 0×0 when
// not exactly computable: missing cup data, non-integral coordinates (the
// greedy H-basis need not be integral), or out-of-int-range entries.
NP_NODISCARD inline ndarray<int> cup_pairing_block(const homology::SimplicialComplex &K, int n, int p, int q, int bp,
                                                   int bq)
{
    const auto empty = ndarray<int>::from_data({0, 0}, std::vector<int>{});
    RationalCohomology rc = rational_cohomology(K);
    if (!rc.ok || n > rc.D || p > rc.D || q > rc.D)
        return empty;
    if (p >= static_cast<int>(rc.cup.size()) || q >= static_cast<int>(rc.cup[p].size()))
        return empty;
    if (static_cast<int>(rc.cup[p][q].size()) != bp)
        return empty;
    if (n >= static_cast<int>(rc.betti.size()) || rc.betti[n] != 1)
        return empty;
    std::vector<int> data(static_cast<std::size_t>(bp) * static_cast<std::size_t>(bq), 0);
    for (int a = 0; a < bp; ++a)
    {
        if (static_cast<int>(rc.cup[p][q][a].size()) != bq)
            return empty;
        for (int b = 0; b < bq; ++b)
        {
            const std::vector<Frac> &coords = rc.cup[p][q][a][b];
            if (coords.size() != 1)
                return empty;
            const Frac &c = coords[0];
            if (!(c.den == bigint("1")))
                return empty; // rational, non-integral basis choice
            long long vll = 0;
            try
            {
                vll = c.num.convert_to<long long>();
            }
            catch (...)
            {
                return empty;
            }
            if (vll > std::numeric_limits<int>::max() || vll < std::numeric_limits<int>::min() ||
                !(c.num == bigint(std::to_string(vll))))
                return empty;
            data[static_cast<std::size_t>(a) * static_cast<std::size_t>(bq) + static_cast<std::size_t>(b)] =
                static_cast<int>(vll);
        }
    }
    return ndarray<int>::from_data({bp, bq}, std::move(data));
}
} // namespace detail

/**
 * @brief Poincaré pairing `H^p × H^{n-p} → Z` via cup + cap fundamental class.
 * For closed oriented n-manifold, pairing is unimodular.
 * Returns matrix `M_{ab}=⟨a⌣b,[M]⟩` as `ndarray<int>` of size `betti_p × betti_{n-p}`.
 *
 * NOTE (honesty audit): an earlier revision returned a hardcoded diagonal-1
 * matrix with no cup evaluation at all. Entries are now read off the real
 * Alexander–Whitney cup table (see detail::cup_pairing_block). Returns an
 * empty 0×0 array when the pairing is not computable exactly here.
 */
NP_NODISCARD inline ndarray<int> poincare_pairing(const homology::SimplicialComplex &K)
{
    const auto empty = ndarray<int>::from_data({0, 0}, std::vector<int>{});
    auto hg = homology::homology_groups(K);
    int n = detail::effective_dim(hg);
    if (n < 0 || hg[n].betti != 1)
        return empty;
    // Try middle pairing first, fallback to H^0×H^n which is always 1×1 for closed
    // manifold
    int half = n / 2;
    int p = half;
    int q = n - p;
    int bp = (p <= n) ? hg[p].betti : 0;
    int bq = (q <= n) ? hg[q].betti : 0;
    if (bp == 0 || bq == 0)
    {
        p = 0;
        q = n;
        bp = hg[p].betti;
        bq = hg[q].betti;
        if (bp == 0 || bq == 0)
            return empty;
    }
    return detail::cup_pairing_block(K, n, p, q, bp, bq);
}

/**
 * @brief Intersection form `Q: H_{n/2} × H_{n/2} → Z` for closed oriented 4k-manifold.
 * Returns `ndarray<int>` `b × b` where `b = betti_{2k}`.
 *
 * NOTE (honesty audit): an earlier revision hardcoded CP2 → [1], S2×S2 →
 * [[0,1],[1,0]], and identity everywhere else, with no cup evaluation.
 * This now reads the middle-dimensional cup block straight from
 * rational_cohomology() (the intersection form IS the H^{2k}×H^{2k} cup
 * pairing under Poincaré duality). Empty 0×0 when not exactly computable.
 */
NP_NODISCARD inline ndarray<int> intersection_form(const homology::SimplicialComplex &K)
{
    const auto empty = ndarray<int>::from_data({0, 0}, std::vector<int>{});
    auto hg = homology::homology_groups(K);
    int n = detail::effective_dim(hg);
    if (n % 4 != 0)
        return empty;
    int mid = n / 2;
    int b = hg[mid].betti;
    if (b == 0)
        return empty;
    if (hg[n].betti != 1)
        return empty;
    // Middle-only: unlike poincare_pairing there is no (0,n) fallback — an
    // H^0×H^n block is not the intersection form.
    return detail::cup_pairing_block(K, n, mid, mid, b, b);
}

/**
 * @brief Künneth for cohomology: H^n(X×Y) ≅ ⊕_{p+q=n} H^p(X)⊗H^q(Y) ⊕ ⊕ Tor.
 * Returns Betti numbers for product.
 */
NP_NODISCARD inline std::vector<int> kunneth_cohomology_betti(const homology::SimplicialComplex &A,
                                                              const homology::SimplicialComplex &B)
{
    auto ca = cohomology_groups(A);
    auto cb = cohomology_groups(B);
    int da = static_cast<int>(ca.size()) - 1, db = static_cast<int>(cb.size()) - 1;
    int D = da + db;
    std::vector<int> out(D + 1, 0);
    for (int i = 0; i <= da; ++i)
        for (int j = 0; j <= db; ++j)
            out[i + j] += ca[i].betti * cb[j].betti;
    return out;
}

/**
 * @brief Universal coefficients short exact sequence data for cohomology.
 */
struct UCT
{
    int betti = 0;
    std::vector<bigint> torsion;
    std::vector<bigint> ext; // Ext(H_{n-1},Z) = torsion_{n-1}
};

NP_NODISCARD inline UCT universal_coefficients(const homology::SimplicialComplex &K, int n)
{
    auto hg = homology::homology_groups(K);
    UCT u;
    if (n < 0 || n >= static_cast<int>(hg.size()))
        return u;
    u.betti = hg[n].betti;
    if (n > 0)
        u.ext = hg[n - 1].torsion;
    u.torsion = u.ext;
    return u;
}

NP_NODISCARD inline std::string cohomology_ring_string(const homology::SimplicialComplex &K)
{
    return cohomology_ring(K).to_string();
}

} // namespace np::cohomology

#endif // NP_COHOMOLOGY_HPP
