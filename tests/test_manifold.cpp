/**
 * @file test_manifold.cpp
 * @brief Tests for manifold (correct name for variety) with logical reasoning.
 */
#include "test_util.hpp"
#include <np/np.hpp>

int main()
{
    using namespace np::manifold;
    using namespace np::homology;

    // ── Sphere S² via manifold ──────────────────────────────────────────────
    {
        auto S2 = np::manifold::make_sphere(2);
        test::check(S2.dimension() == 2, "manifold S2 dim");
        test::check(S2.is_orientable(), "S2 orientable");
        test::check(S2.is_compact(), "S2 compact");
        test::check(S2.is_simply_connected(), "S2 simply connected");
        auto hg = S2.homology();
        test::check(hg[0].betti == 1 && hg[2].betti == 1, "S2 homology");
        test::check(S2.de_rham(2).betti == 1, "S2 de Rham H2=R");
        auto rep = S2.check_logical_consistency();
        test::check(rep.ok, "S2 logical consistency");
    }

    // ── Sphere S^n returns R for n==dim ───────────────────────────────────
    for (int n = 0; n <= 3; ++n)
    {
        auto Sn = np::manifold::make_sphere(n);
        auto dr_n = Sn.de_rham(n);
        test::check(dr_n.betti == 1, "S^n de Rham Hn=R");
        auto dr0 = Sn.de_rham(0);
        test::check(dr0.betti == 1, "S^n de Rham H0=R");
        if (n >= 1)
            test::check(Sn.de_rham(1).betti == (n == 1 ? 1 : 0), "S^n de Rham H1");
        // Over Z, H_n = Z
        test::check(Sn.homology(n).betti == 1, "S^n homology Hn=Z");
    }

    // ── Torus ───────────────────────────────────────────────────────────────
    {
        auto T2 = make_torus(2);
        test::check(T2.is_orientable(), "T2 orientable");
        test::check(T2.homology(1).betti == 2, "T2 H1=Z^2");
        test::check(T2.check_logical_consistency().ok, "T2 consistent");
    }

    // ── Projective ─────────────────────────────────────────────────────────
    {
        auto CP1 = make_complex_projective(1);
        test::check(CP1.is_orientable(), "CP1 orientable");
        test::check(CP1.is_kahler(), "CP1 Kahler");
        auto RP2 = make_real_projective(2);
        test::check(!RP2.is_orientable(), "RP2 non-orientable");
    }

    // ── Variety alias (backward compat) ───────────────────────────────────
    {
        auto S2_v = np::variety::sphere_ptr(2);
        test::check(S2_v->dimension() == 2, "variety alias sphere_ptr");
        auto S2m = np::manifold::make_sphere(2);
        auto S2v = np::variety::SphereVariety(2);
        test::check(S2v.dimension() == 2, "variety::SphereVariety alias");
        (void)S2m;
    }

    // ── Affine scheme ──────────────────────────────────────────────────────
    {
        AffineScheme circle{.equations = {"x^2 + y^2 - 1"}, .ambient_dim = 2};
        test::check(circle.krull_dimension() == 1, "circle Krull dim 1");
        test::check(circle.is_smooth(), "circle smooth");
        test::check(circle.is_irreducible(), "circle irreducible");
    }

    // ── Homotopy via manifold ──────────────────────────────────────────────
    {
        auto S1 = np::manifold::make_sphere(1);
        auto S2 = np::manifold::make_sphere(2);
        test::check(!is_homotopy_equivalent(S1, S2), "S1 != S2 homotopy");
        test::check(is_homotopy_equivalent(S1, S1), "S1 homotopy self");
    }

    // ── Rational cup product (cochain-level, not pattern tables) ──────────
    {
        using namespace np::cohomology;
        auto T2 = np::manifold::torus_complex(2);
        auto RT2 = cohomology_ring(T2);
        test::check(!RT2.inconclusive, "T2 cup conclusive");
        test::check(cup_pairing_rank(T2, 1, 1) == 1, "T2 cup rank 1");
        bool t2_nonzero = false;
        for (int a = 0; a < 2; ++a)
        {
            for (int b = 0; b < 2; ++b)
            {
                if (cup_product(T2, 1, 1, a, b) >= 0)
                {
                    t2_nonzero = true;
                }
            }
        }
        test::check(t2_nonzero, "T2 some H1 cup nonzero");
        // Wedge S¹∨S¹∨S²: same Betti [1,2,1] and Euler 0 as T², trivial cup.
        SimplicialComplex W({
            {{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}},
            {{0, 1}, {1, 2}, {0, 2}, {0, 3}, {3, 4}, {0, 4}, {0, 5}, {0, 6}, {0, 7}, {5, 6}, {5, 7}, {6, 7}},
            {{0, 5, 6}, {0, 5, 7}, {0, 6, 7}, {5, 6, 7}},
        });
        auto bettiW = betti_numbers(W);
        test::check(bettiW.size() == 3 && bettiW[0] == 1 && bettiW[1] == 2 && bettiW[2] == 1, "wedge Betti [1,2,1]");
        auto RW = cohomology_ring(W);
        test::check(!RW.inconclusive, "wedge cup conclusive");
        test::check(cup_pairing_rank(W, 1, 1) == 0, "wedge cup rank 0");
        bool wedge_allzero = true;
        for (int a = 0; a < 2; ++a)
        {
            for (int b = 0; b < 2; ++b)
            {
                if (cup_product(W, 1, 1, a, b) != -1)
                {
                    wedge_allzero = false;
                }
            }
        }
        test::check(wedge_allzero, "wedge H1 cups all zero");
        // Same homology, different ring: conclusively not equivalent
        // (previously provisional-true).
        auto r = np::homotopy::is_homotopy_equivalent(T2, W);
        test::check(!r.equivalent && !r.inconclusive, "T2 vs wedge distinguished by cup");
        // S² self: agreement stays provisional (Whitehead needs a map).
        auto S2 = np::manifold::sphere_complex(2);
        test::check(cup_product(S2, 0, 2, 0, 0) == 0, "S2 unit cup");
        auto rs = np::homotopy::is_homotopy_equivalent(S2, S2);
        test::check(rs.equivalent && rs.inconclusive, "S2 self provisional");
        // Circle: no H², cup rank 0, still conclusive.
        auto S1 = np::manifold::sphere_complex(1);
        test::check(cup_pairing_rank(S1, 1, 1) == 0, "S1 cup rank 0");
        test::check(!cohomology_ring(S1).inconclusive, "S1 cup conclusive");
    }

    // ── AnyManifold variant ────────────────────────────────────────────────
    {
        AnyManifold v = np::manifold::make_sphere(2);
        test::check(std::visit([](auto &x) { return x.dimension(); }, v) == 2, "AnyManifold visit");
        test::check(name(v) == "S^2", "AnyManifold name");
    }

    return test::failures() ? 1 : 0;
}
