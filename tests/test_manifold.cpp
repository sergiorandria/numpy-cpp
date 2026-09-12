/**
 * @file test_manifold.cpp
 * @brief Tests for manifold (correct name for variety) with logical reasoning.
 */
#include "test_util.hpp"
#include <cmath>
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
        // S^0 is two points: H^0 = R^2; all other S^n have H^0 = R.
        test::check(dr_n.betti == (n == 0 ? 2 : 1), "S^n de Rham Hn");
        auto dr0 = Sn.de_rham(0);
        test::check(dr0.betti == (n == 0 ? 2 : 1), "S^n de Rham H0");
        if (n >= 1)
            test::check(Sn.de_rham(1).betti == (n == 1 ? 1 : 0), "S^n de Rham H1");
        // Over Z: H_n = Z (n>=1), H_0(S^0) = Z^2.
        test::check(Sn.homology(n).betti == (n == 0 ? 2 : 1), "S^n homology Hn");
        test::check(Sn.check_logical_consistency().ok, "S^n logical consistency");
    }
    // ── Sphere volumes: 2*pi^{(n+1)/2}/Gamma((n+1)/2) ────────────────────
    {
        auto S2 = np::manifold::make_sphere(2);
        test::check(std::abs(S2.volume() - 4.0 * 3.141592653589793) < 1e-9, "S2 volume 4pi");
        auto S3 = np::manifold::make_sphere(3);
        test::check(std::abs(S3.volume() - 2.0 * 3.141592653589793 * 3.141592653589793) < 1e-9, "S3 volume 2pi^2");
        test::check(std::abs(S3.scalar_curvature({}) - 6.0) < 1e-12, "S3 scalar curvature 6");
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
        test::check(circle.is_smooth_heuristic(), "circle smooth");
        test::check(circle.is_irreducible_heuristic(), "circle irreducible");
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

    // ── Euclidean space ────────────────────────────────────────────────────
    {
        auto R3 = np::manifold::make_euclidean(3);
        test::check(!R3.is_compact() && R3.is_complete(), "R3 non-compact complete");
        test::check(R3.is_simply_connected(), "R3 simply connected");
        test::check(R3.homology(0).betti == 1 && R3.homology(1).betti == 0, "R3 homology");
        test::check(R3.check_logical_consistency().ok, "R3 consistent");
        auto p = np::manifold::sphere(2);
        test::check(p->dimension() == 2, "manifold::sphere pointer factory");
    }

    // ── Genus-g surfaces ───────────────────────────────────────────────────
    {
        auto Sg2 = np::manifold::make_genus_g_surface(2);
        test::check(Sg2.homology(1).betti == 4, "Sigma2 H1=Z^4");
        test::check(Sg2.euler_characteristic() == -2, "Sigma2 Euler -2");
        test::check(Sg2.is_orientable() && !Sg2.is_simply_connected(), "Sigma2 orientable non-sc");
        test::check(Sg2.homotopy(2).rank == 0 && !Sg2.homotopy(2).inconclusive, "Sigma2 aspherical");
        test::check(Sg2.check_logical_consistency().ok, "Sigma2 consistent");
        auto S0 = np::manifold::make_genus_g_surface(0);
        test::check(S0.is_simply_connected() && S0.euler_characteristic() == 2, "Sigma0 = S2");
    }

    // ── Lens spaces ────────────────────────────────────────────────────────
    {
        auto L = np::manifold::make_lens_space(5, 1);
        test::check(L.homology(1).torsion.size() == 1, "L(5;1) H1 torsion");
        test::check(L.homology(3).betti == 1, "L H3=Z");
        test::check(L.de_rham(1).betti == 0 && L.de_rham(3).betti == 1, "L de Rham kills torsion");
        test::check(L.homotopy(2).rank == 0 && L.homotopy(3).rank == 1, "L pi2/pi3 from S3 cover");
        test::check(L.check_logical_consistency().ok, "L consistent");
    }

    // ── Connected sum ──────────────────────────────────────────────────────
    {
        auto sum = np::manifold::make_connected_sum(np::manifold::torus(), np::manifold::torus());
        test::check(sum.dimension() == 2, "T2#T2 dim");
        test::check(sum.homology(1).betti == 4, "T2#T2 H1=Z^4");
        test::check(sum.euler_characteristic() == -2, "T2#T2 Euler -2");
        test::check(sum.is_orientable(), "T2#T2 orientable");
        test::check(sum.check_logical_consistency().ok, "T2#T2 consistent");
    }

    // ── Product Künneth torsion: RP^2 x S^1 ────────────────────────────────
    {
        auto prod = np::manifold::make_product(np::manifold::real_projective(2), np::manifold::sphere(1));
        // H_2 has Tor(H_1(RP2),H_1(S1)) = Tor(Z/2,Z) = 0; H_1 = Z + Z/2.
        test::check(prod.homology(1).betti == 1, "RP2xS1 H1 betti 1");
        test::check(!prod.homology(1).torsion.empty(), "RP2xS1 H1 torsion Z/2");
        auto prod2 = np::manifold::make_product(np::manifold::real_projective(2), np::manifold::real_projective(2));
        // Tor(Z/2,Z/2) = Z/2 contributes to H_2.
        test::check(!prod2.homology(2).torsion.empty(), "RP2xRP2 H2 Tor torsion");
        test::check(prod.check_logical_consistency().ok, "RP2xS1 consistent");
    }

    // ── AnyManifold variant ────────────────────────────────────────────────
    {
        AnyManifold v = np::manifold::make_sphere(2);
        test::check(std::visit([](auto &x) { return x.dimension(); }, v) == 2, "AnyManifold visit");
        test::check(name(v) == "S^2", "AnyManifold name");
        AnyManifold w = np::manifold::make_lens_space(3, 1);
        test::check(np::manifold::euler_characteristic(w) == 0, "AnyManifold lens Euler");
    }

    // ── Simplicial-vs-authoritative cross-check ──────────────────────────
    // to_simplicial() MUST agree with homology() (betti + torsion) or be a
    // documented placeholder (see AbstractManifold::to_simplicial contract).
    // NOTE: betti-only comparison would miss Lens torsion, so torsion is
    // compared too. If you implement a faithful triangulation, move its row
    // to the agree-table below.
    {
        auto agrees = [](const auto &M) {
            auto hg = M.homology();
            auto sc = M.to_simplicial();
            auto bs = np::homology::betti_numbers(sc);
            auto hs = np::homology::homology_groups(sc);
            // Lengths may differ by trailing-zero padding; compare degree
            // by degree with missing entries treated as (0, no torsion).
            const size_t n = std::max({hg.size(), bs.size(), hs.size()});
            for (size_t k = 0; k < n; ++k)
            {
                const int bb = k < bs.size() ? bs[k] : 0;
                const int hb = k < hg.size() ? hg[k].betti : 0;
                if (bb != hb)
                    return false;
                const bool st = k < hs.size() ? !hs[k].torsion.empty() : false;
                const bool ht = k < hg.size() ? !hg[k].torsion.empty() : false;
                if (st != ht)
                    return false;
                if (st && ht && !(hs[k].torsion == hg[k].torsion))
                    return false;
            }
            return true;
        };
        test::check(agrees(np::manifold::make_sphere(2)), "simplicial agrees S2");
        test::check(agrees(np::manifold::TorusManifold(2)), "simplicial agrees T2");
        test::check(agrees(np::manifold::make_sphere(1)), "simplicial agrees S1");
        // Documented placeholders (disagreement is known, not a regression):
        test::check(!agrees(np::manifold::make_lens_space(5, 1)), "placeholder Lens(5) differs");
        test::check(!agrees(np::manifold::GenusGSurfaceManifold(2)), "placeholder genus-2 differs");
        test::check(!agrees(np::manifold::ProjectiveManifold("C", 2)), "placeholder CP2 differs");
        test::check(!agrees(np::manifold::TorusManifold(3)), "placeholder T3 differs");
        auto cs = np::manifold::make_connected_sum(std::make_unique<np::manifold::TorusManifold>(2),
                                                   std::make_unique<np::manifold::TorusManifold>(2));
        test::check(!agrees(cs), "placeholder connected-sum differs");
    }

    return test::failures() ? 1 : 0;
}
