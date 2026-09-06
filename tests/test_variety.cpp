/**
 * @file test_variety.cpp
 * @brief Tests for abstract varieties (sphere, torus, projective) with homology/homotopy/de Rham.
 */
#include "test_util.hpp"
#include <np/np.hpp>

int main()
{
    using namespace np::variety;
    using namespace np::homology;
    using namespace np::homotopy;

    // ── Sphere S² ─────────────────────────────────────────────────────────
    {
        auto S2 = sphere(2);
        test::check(S2->name() == "S^2", "sphere name");
        test::check(S2->dimension() == 2, "sphere dim 2");
        test::check(S2->euler_characteristic() == 2, "sphere Euler 2");
        auto hg = S2->homology();
        test::check(hg.size() == 3 && hg[0].betti == 1 && hg[1].betti == 0 && hg[2].betti == 1,
                    "sphere homology [1,0,1]");
        test::check(S2->homology(0).betti == 1 && S2->homology(2).betti == 1, "sphere homology 0,2");
        test::check(S2->homology(1).betti == 0, "sphere H1 0");
        auto dr0 = S2->de_rham(0);
        auto dr2 = S2->de_rham(2);
        auto dr1 = S2->de_rham(1);
        test::check(dr0.betti == 1 && dr2.betti == 1 && dr1.betti == 0, "sphere de Rham R in 0,n");
        // Over R, H_n = R
        test::check(dr2.betti == 1, "sphere de Rham H2 = R");
        auto pi2 = S2->homotopy(2);
        test::check(pi2.rank == 1 && !pi2.inconclusive, "pi2(S2)=Z");
        // Simplicial
        auto K = S2->to_simplicial();
        auto betti = betti_numbers(K);
        test::check(betti[0] == 1 && betti[1] == 0 && betti[2] == 1, "sphere simplicial Betti");
    }

    // ── Sphere S^n general ────────────────────────────────────────────────
    for (int n = 0; n <= 3; ++n)
    {
        auto Sn = sphere(n);
        auto hg = Sn->homology();
        // H_0 = Z, H_n = Z, others 0
        test::check(hg[0].betti == 1, "S^n H0=Z");
        if (n >= 1)
            test::check(hg[n].betti == 1, "S^n Hn=Z");
        for (int k = 1; k < n; ++k)
            test::check(hg[k].betti == 0, "S^n intermediate 0");
        // de Rham
        auto dr = Sn->de_rham(n);
        test::check(dr.betti == 1, "S^n de Rham Hn=R");
        auto dr0 = Sn->de_rham(0);
        test::check(dr0.betti == 1, "S^n de Rham H0=R");
        if (n >= 1)
            test::check(Sn->de_rham(1).betti == (n == 1 ? 1 : 0), "S^n de Rham H1");
    }

    // ── Torus T² ──────────────────────────────────────────────────────────
    {
        auto T2 = torus(2);
        test::check(T2->euler_characteristic() == 0, "torus Euler 0");
        auto hg = T2->homology();
        test::check(hg[0].betti == 1 && hg[1].betti == 2 && hg[2].betti == 1, "torus homology [1,2,1]");
        test::check(T2->homotopy(1).rank == 2, "pi1(T2)=Z^2");
    }

    // ── Projective spaces ─────────────────────────────────────────────────
    {
        auto RP2 = real_projective(2);
        test::check(RP2->euler_characteristic() == 1, "RP2 Euler 1");
        auto hg = RP2->homology();
        // Over Z, H1=Z/2, H2=0 for RP2
        test::check(hg[0].betti == 1, "RP2 H0");
        auto CP1 = complex_projective(1); // CP1 ≅ S2
        test::check(CP1->homology(0).betti == 1 && CP1->homology(2).betti == 1, "CP1 homology");
        test::check(CP1->euler_characteristic() == 2, "CP1 Euler 2");
    }

    // ── Wedge ─────────────────────────────────────────────────────────────
    {
        std::vector<std::unique_ptr<AbstractVariety>> parts;
        parts.push_back(sphere(1));
        parts.push_back(sphere(1));
        auto wedge = std::make_unique<WedgeVariety>(std::move(parts));
        test::check(wedge->homology(1).betti == 2, "wedge S1∨S1 H1=Z^2");
        test::check(wedge->euler_characteristic() == -1, "wedge Euler -1");
    }

    return test::failures() ? 1 : 0;
}
