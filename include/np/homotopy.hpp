/**
 * @file homotopy.hpp
 * @brief Homotopy equivalence and related invariants for simplicial complexes.
 *
 * Provides `np::homotopy` routines that decide homotopy equivalence for
 * finite simplicial complexes via computable invariants:
 *   - `is_simply_connected`, `is_contractible`, `is_aspherical`
 *   - `is_homotopy_equivalent` (Betti + Euler + H₁ torsion + rational cup,
 *     Whitehead with the inducing-map caveat)
 *   - `fundamental_group_abelianization` (H₁)
 *   - `homotopy_group` (π₁ via H₁, higher via Hurewicz/aspherical)
 *
 * The general homotopy equivalence problem is undecidable; these routines
 * implement necessary invariants that are sufficient for many classical
 * examples (spheres, tori, wedges, graphs) and otherwise return
 * `inconclusive=true` conservatively.
 *
 * Soundness note: Whitehead's theorem needs a *map* inducing the homology
 * isomorphism, not just abstractly isomorphic homology (CP² vs S²∨S⁴ is the
 * textbook counterexample). The simply-connected branch therefore compares
 * rational cup products and stays provisional on agreement instead of
 * claiming a conclusive equivalence.
 *
 * Improvements over previous stub:
 *   - Graphs (1-dim) are aspherical: π_{≥2}=0 conclusively (universal cover is a tree).
 *   - Whitehead: simply-connected + homology iso ⇒ equivalent; otherwise
 *     non-simply-connected higher dims are inconclusive unless both 1-skeleta.
 *   - `homotopy_group` handles aspherical case and Hurewicz correctly.
 *
 * Reference: Hatcher, *Algebraic Topology* Ch.1-4; Whitehead theorem
 * (weak homotopy equivalence + CW ⇒ homotopy equivalence).
 *
 * @author Sergio Randriamihoatra (sergiorandriamihoatra@gmail.com)
 */
#ifndef NP_HOMOTOPY_HPP
#define NP_HOMOTOPY_HPP

#include <string>
#include <vector>

#include "api_macros.hpp"
#include "cohomology.hpp"
#include "homology.hpp"

namespace np::homotopy
{

namespace detail
{
/**
 * @brief True unless both rings are conclusive with different rational cup
 * pairing ranks. Basis-independent comparison; unknown (inconclusive ring,
 * oversized complex, malformed input) counts as agree, keeping the verdict
 * provisional rather than wrong.
 */
NP_NODISCARD inline bool rational_cups_agree(const homology::SimplicialComplex &A, const homology::SimplicialComplex &B)
{
    try
    {
        const auto RA = cohomology::cohomology_ring(A);
        const auto RB = cohomology::cohomology_ring(B);
        if (RA.inconclusive || RB.inconclusive)
        {
            return true;
        }
        if (RA.groups.size() != RB.groups.size())
        {
            return false;
        }
        const int D = static_cast<int>(RA.groups.size()) - 1;
        for (int p = 0; p <= D; ++p)
        {
            for (int q = 0; q <= D; ++q)
            {
                if (cohomology::cup_pairing_rank(A, p, q) != cohomology::cup_pairing_rank(B, p, q))
                {
                    return false;
                }
            }
        }
        return true;
    }
    catch (const std::invalid_argument &)
    {
        return true; // malformed (unclosed) input: stay provisional, fail loud elsewhere
    }
}
} // namespace detail

struct HomotopyResult
{
    bool equivalent = false;
    bool inconclusive = false;
    std::string reason;
};

NP_NODISCARD inline bool is_simply_connected(const homology::SimplicialComplex &K)
{
    auto hg = homology::homology_groups(K);
    if (hg.size() <= 1)
        return true;
    return hg[1].betti == 0 && hg[1].torsion.empty();
}

NP_NODISCARD inline bool is_simply_connected(const std::vector<np::ndarray<int>> &bms)
{
    auto hg = homology::homology_groups(bms);
    if (hg.size() <= 1)
        return true;
    return hg[1].betti == 0 && hg[1].torsion.empty();
}

NP_NODISCARD inline bool is_aspherical_graph(const homology::SimplicialComplex &K)
{
    return K.dim() <= 1;
}

NP_NODISCARD inline bool is_contractible(const homology::SimplicialComplex &K)
{
    auto betti = homology::betti_numbers(K);
    if (betti.empty() || betti[0] != 1)
        return false;
    for (size_t i = 1; i < betti.size(); ++i)
        if (betti[i] != 0)
            return false;
    auto hg = homology::homology_groups(K);
    for (auto &g : hg)
        if (!g.torsion.empty())
            return false;
    if (!is_simply_connected(K))
        return false;
    return homology::euler_characteristic(K) == 1;
}

NP_NODISCARD inline std::vector<homology::HomologyGroup> fundamental_group_abelianization(
    const homology::SimplicialComplex &K)
{
    auto hg = homology::homology_groups(K);
    if (hg.size() <= 1)
        return {};
    return {hg[1]};
}

NP_NODISCARD inline std::vector<homology::HomologyGroup> fundamental_group_abelianization(
    const std::vector<ndarray<int>> &bms)
{
    auto hg = homology::homology_groups(bms);
    if (hg.size() <= 1)
        return {};
    return {hg[1]};
}

/**
 * @brief Homotopy equivalence via Whitehead + computable invariants.
 *
 * Checks:
 *   1. `betti_numbers` equality (over Q)
 *   2. `euler_characteristic` equality
 *   3. `H₁` torsion equality (abelianization of π₁)
 *   4. Rational cup-product agreement (conclusive `false` on mismatch).
 *   5. If both simply connected and 1-4 hold: provisional `true`
 *      (Whitehead needs an inducing map, not just abstract iso).
 *   6. If both graphs (dim≤1) and 1-3 hold, homology determines homotopy.
 * Otherwise returns `inconclusive=true` (higher invariants needed).
 */
NP_NODISCARD inline HomotopyResult is_homotopy_equivalent(const homology::SimplicialComplex &A,
                                                          const homology::SimplicialComplex &B)
{
    auto bettiA = homology::betti_numbers(A);
    auto bettiB = homology::betti_numbers(B);
    if (bettiA != bettiB)
        return {false, false, "Betti numbers differ"};

    int eA = homology::euler_characteristic(A);
    int eB = homology::euler_characteristic(B);
    if (eA != eB)
        return {false, false, "Euler characteristic differs"};

    auto hA = homology::homology_groups(A);
    auto hB = homology::homology_groups(B);
    std::vector<np::bigint> torsA, torsB;
    if (hA.size() > 1)
        torsA = hA[1].torsion;
    if (hB.size() > 1)
        torsB = hB[1].torsion;
    if (torsA != torsB)
        return {false, false, "H1 torsion differs"};

    bool scA = is_simply_connected(A);
    bool scB = is_simply_connected(B);
    if (scA != scB)
        return {false, false, "One simply connected, other not"};

    if (scA && scB)
    {
        if (!detail::rational_cups_agree(A, B))
        {
            return {false, false, "Rational cup products differ"};
        }
        return {true, true, "Simply connected + homology/cup iso; Whitehead needs an inducing map: provisional"};
    }

    // Both non-simply connected
    bool graphA = (A.dim() <= 1);
    bool graphB = (B.dim() <= 1);
    if (graphA && graphB)
    {
        return {true, false, "Graphs: homology determines homotopy (wedge of circles)"};
    }
    if (graphA != graphB)
    {
        return {false, false, "One graph, other not: not homotopy equivalent"};
    }
    // Higher-dimensional non-simply connected: homology iso is necessary but not
    // sufficient (e.g., lens spaces). The rational cup product is a further
    // necessary invariant: mismatch is conclusive, agreement stays provisional.
    if (!detail::rational_cups_agree(A, B))
    {
        return {false, false, "Rational cup products differ"};
    }
    return {true, true,
            "Same H₁+Betti+cup but non-simply connected higher dims: provisional (need π₂, torsion "
            "pairing)"};
}

NP_NODISCARD inline HomotopyResult is_homotopy_equivalent(const std::vector<ndarray<int>> &bmsA,
                                                          const std::vector<ndarray<int>> &bmsB)
{
    auto bettiA = homology::betti_numbers(bmsA);
    auto bettiB = homology::betti_numbers(bmsB);
    if (bettiA != bettiB)
        return {false, false, "Betti numbers differ"};
    int eA = homology::euler_characteristic(bmsA);
    int eB = homology::euler_characteristic(bmsB);
    if (eA != eB)
        return {false, false, "Euler characteristic differs"};
    auto hA = homology::homology_groups(bmsA);
    auto hB = homology::homology_groups(bmsB);
    std::vector<bigint> torsA, torsB;
    if (hA.size() > 1)
        torsA = hA[1].torsion;
    if (hB.size() > 1)
        torsB = hB[1].torsion;
    if (torsA != torsB)
        return {false, false, "H1 torsion differs"};
    bool scA = is_simply_connected(bmsA);
    bool scB = is_simply_connected(bmsB);
    if (scA != scB)
        return {false, false, "One simply connected, other not"};
    if (scA && scB)
        // NOTE (honesty audit): an earlier revision returned conclusive-true
        // here, stronger than the simplicial overload (which stays
        // provisional pending cup data the bms form cannot even carry).
        // Mirror it: equivalent on current evidence, inconclusive overall.
        return {true, true, "Simply connected + homology iso; provisional (no cup data on bms input)"};
    int dimA = static_cast<int>(bmsA.size()) - 1;
    int dimB = static_cast<int>(bmsB.size()) - 1;
    bool graphA = (dimA <= 1);
    bool graphB = (dimB <= 1);
    if (graphA && graphB)
        return {true, false, "Graphs: homology determines"};
    if (graphA != graphB)
        return {false, false, "One graph, other not"};
    return {true, true, "Non-simply connected higher dims provisional (need cup product)"};
}

/**
 * @brief Homotopy group π_n (n=1 via H₁, n>1 via Hurewicz/aspherical).
 *
 * For `n==1` returns `H₁` (abelianization). For `n>1`:
 *   - if aspherical graph (dim≤1) ⇒ π_{≥2}=0 conclusively,
 *   - else if simply connected and lower homology vanishes ⇒ Hurewicz π_n≅H_n,
 *   - else inconclusive.
 */
struct HomotopyGroup
{
    int rank = 0;
    std::vector<bigint> torsion;
    bool inconclusive = false;
    std::string to_string() const
    {
        if (inconclusive)
            return "inconclusive";
        std::string s = "Z^" + std::to_string(rank);
        for (auto &t : torsion)
            s += " + Z/" + t.convert_to<std::string>() + "Z";
        return s;
    }
};

NP_NODISCARD inline HomotopyGroup homotopy_group(const homology::SimplicialComplex &K, int n)
{
    if (n <= 0)
        return {0, {}, true};
    auto hg = homology::homology_groups(K);
    if (n >= static_cast<int>(hg.size()))
    {
        // Beyond homology range: if aspherical graph, still 0
        if (K.dim() <= 1 && n >= 2)
            return {0, {}, false};
        // NOTE (honesty audit): an earlier revision returned conclusive 0
        // here, but e.g. pi_5(S^2) = Z/2 lives beyond any homology range.
        // Hurewicz does not apply, so this is unknown, not zero.
        return {0, {}, true};
    }
    if (n == 1)
        return {hg[1].betti, hg[1].torsion, false};
    // n >=2
    if (K.dim() <= 1)
    {
        // Graphs are K(G,1): π_{≥2}=0
        return {0, {}, false};
    }
    if (!is_simply_connected(K))
        return {0, {}, true};
    for (int i = 1; i < n; ++i)
        if (hg[i].betti != 0 || !hg[i].torsion.empty())
            return {0, {}, true};
    return {hg[n].betti, hg[n].torsion, false};
}

NP_NODISCARD inline HomotopyGroup homotopy_group(const std::vector<ndarray<int>> &bms, int n)
{
    if (n <= 0)
        return {0, {}, true};
    auto hg = homology::homology_groups(bms);
    if (n >= static_cast<int>(hg.size()))
    {
        int dim = static_cast<int>(bms.size()) - 1;
        if (dim <= 1 && n >= 2)
            return {0, {}, false};
        return {0, {}, true};
    }
    if (n == 1)
        return {hg[1].betti, hg[1].torsion, false};
    int dim = static_cast<int>(bms.size()) - 1;
    if (dim <= 1)
        return {0, {}, false};
    if (!is_simply_connected(bms))
        return {0, {}, true};
    for (int i = 1; i < n; ++i)
        if (hg[i].betti != 0 || !hg[i].torsion.empty())
            return {0, {}, true};
    return {hg[n].betti, hg[n].torsion, false};
}

} // namespace np::homotopy

#endif // NP_HOMOTOPY_HPP
