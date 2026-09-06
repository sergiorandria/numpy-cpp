/**
 * @file test_abstract.cpp
 * @brief Tests for homology, homotopy, modular forms.
 */
#include "test_util.hpp"
#include <np/np.hpp>

int main()
{
    using namespace np;
    using namespace np::homology;
    using namespace np::homotopy;
    using namespace np::modular;

    // ── Homology ──────────────────────────────────────────────────────────
    {
        // SNF 2x2
        ndarray<int> A = ndarray<int>::from_data({2, 2}, std::vector<int>{2, 4, 6, 8});
        auto diag = smith_normal_form(A);
        test::check(diag.size() == 2, "SNF size 2");
        // For [[2,4],[6,8]] det=-8, gcd=2 => diag [2,4]
        test::check(diag[0] == bigint(2) && diag[1] == bigint(4), "SNF 2x2 values");
    }
    {
        // Betti: circle
        auto circ = circle_complex();
        auto betti = betti_numbers(circ);
        test::check(betti.size() >= 2 && betti[0] == 1 && betti[1] == 1, "Betti circle [1,1]");
        test::check(euler_characteristic(circ) == 0, "Euler circle 0");
        auto hg = homology_groups(circ);
        test::check(hg[0].betti == 1 && hg[1].betti == 1, "homology circle");
    }
    {
        // Betti: sphere tetrahedron
        auto sph = sphere_tetrahedron();
        auto betti = betti_numbers(sph);
        test::check(betti.size() >= 3 && betti[0] == 1 && betti[1] == 0 && betti[2] == 1, "Betti sphere [1,0,1]");
        test::check(euler_characteristic(sph) == 2, "Euler sphere 2");
    }
    {
        // Point
        SimplicialComplex pt{{{{0}}, {}, {}}};
        // Actually need proper: 1 vertex, no edges
        pt.simplices = {{{0}}, {}, {}};
        auto betti = betti_numbers(pt);
        test::check(betti[0] == 1, "Betti point [1]");
        test::check(euler_characteristic(pt) == 1, "Euler point 1");
    }

    // ── Homotopy ──────────────────────────────────────────────────────────
    {
        auto circ = circle_complex();
        auto sph = sphere_tetrahedron();
        auto pt = SimplicialComplex{{{{0}}, {}, {}}};
        pt.simplices = {{{0}}, {}, {}};

        test::check(is_simply_connected(sph), "sphere simply connected");
        test::check(!is_simply_connected(circ), "circle not simply connected");
        test::check(is_contractible(pt), "point contractible");
        test::check(!is_contractible(circ), "circle not contractible");

        auto r1 = is_homotopy_equivalent(circ, circ);
        test::check(r1.equivalent, "circle homotopy self");
        auto r2 = is_homotopy_equivalent(circ, sph);
        test::check(!r2.equivalent, "circle vs sphere not homotopy");

        auto hg = homotopy_group(circ, 1);
        test::check(hg.rank == 1, "pi1 circle rank 1");
        auto hg2 = homotopy_group(sph, 2);
        test::check(hg2.rank == 1, "pi2 sphere rank 1");
    }

    // ── Modular forms ─────────────────────────────────────────────────────
    {
        // sigma
        test::check(sigma(1, 6) == bigint(12), "sigma 1,6 =1+2+3+6=12");
        test::check(sigma(3, 2) == bigint(9), "sigma 3,2=1+8=9");
        // Bernoulli
        auto [num, den] = bernoulli(4);
        test::check(num == bigint(-1) && den == bigint(30), "bernoulli 4");
        // Eisenstein E4
        auto E4 = eisenstein_series(4, 4);
        test::check(E4.at(0) == bigint(1) && E4.at(1) == bigint(240) && E4.at(2) == bigint(2160) &&
                        E4.at(3) == bigint(6720),
                    "Eisenstein E4");
        auto E6 = eisenstein_series(6, 3);
        test::check(E6.at(0) == bigint(1) && E6.at(1) == bigint(-504) && E6.at(2) == bigint(-16632), "Eisenstein E6");
        // Hecke
        auto a = eisenstein_series(4, 10);
        auto Tp = hecke_operator(a, 4, 2);
        // For eigenform, Tp should be eigen: check first few
        // E4 is eigen with eigenvalue sigma3(p)=1+p^3
        bigint eigen = bigint(1) + bigint(8); // 1+2^3=9 for p=2? Wait sigma3(2)=9, but eigenvalue for E4 is sigma3(p)=9
        // Actually Hecke eigenvalue for E4 is sigma3(p)=9
        // Check Tp[0] == 9 * a[0]? a[0]=1, Tp[0]=1+8=9
        test::check(Tp.at(0) == a.at(0) * (bigint(1) + bigint(8)), "hecke E4 p=2");
        // Dedekind eta & j
        std::complex<double> tau(0, 1); // i
        auto eta = dedekind_eta(tau, 20);
        test::check(std::abs(eta) > 0, "dedekind_eta non-zero");
        auto jser = j_invariant_series(3);
        test::check(jser[0] == 744 && jser[1] == 196884, "j-invariant series");
    }

    return test::failures() ? 1 : 0;
}
