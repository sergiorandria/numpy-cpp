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
        // SNF: torsion and divisibility on larger shapes (Kannan–Bachem path).
        auto snf = [](std::vector<int> shape, std::vector<int> vals) {
            return smith_normal_form(ndarray<int>::from_data(shape, vals));
        };
        auto d1 = snf({3, 3}, {2, 4, 4, -2, -3, 1, 4, 2, 4});
        test::check(d1.size() == 3 && d1[0] == bigint(1) && d1[1] == bigint(2) && d1[2] == bigint(26),
                    "SNF 3x3 torsion [1,2,26]");
        auto d2 = snf({2, 3}, {3, 0, 0, 0, 5, 0});
        test::check(d2.size() == 2 && d2[0] == bigint(1) && d2[1] == bigint(15), "SNF 2x3 [1,15]");
        auto d3 = snf({4, 3}, {3, -1, 4, 0, 0, 3, 2, 4, 3, 2, -1, 4});
        test::check(d3.size() == 3, "SNF 4x3 size");
        bool div_ok = true;
        bigint prev = 1;
        for (auto &v : d3)
        {
            if (v == 0)
            {
                break;
            }
            if (v % prev != 0)
            {
                div_ok = false;
            }
            prev = v;
        }
        test::check(div_ok, "SNF divisibility chain");
        // RP² has H₁ = Z/2: boundary ∂₂ = [2] gives torsion 2.
        auto drp = snf({1, 1}, {2});
        test::check(drp.size() == 1 && drp[0] == bigint(2), "SNF RP2 torsion [2]");
        // Zero matrix stays zero.
        auto dz = snf({2, 2}, {0, 0, 0, 0});
        test::check(dz[0] == bigint(0) && dz[1] == bigint(0), "SNF zero matrix");
        // Perf smoke: 14x14 dense-ish completes (minors would enumerate ~10^8).
        std::vector<int> big(14 * 14);
        unsigned long long st = 42;
        for (auto &v : big)
        {
            st = st * 6364136223846793005ULL + 1442695040888963407ULL;
            v = static_cast<int>((st >> 33) % 7) - 3;
        }
        auto db = snf({14, 14}, big);
        test::check(db.size() == 14, "SNF 14x14 completes");
        bool bdiv = true;
        bigint bp = 1;
        for (auto &v : db)
        {
            if (v == 0)
            {
                break;
            }
            if (v % bp != 0)
            {
                bdiv = false;
            }
            bp = v;
        }
        test::check(bdiv, "SNF 14x14 divisibility");
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
