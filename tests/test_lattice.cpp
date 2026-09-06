/**
 * @file test_lattice.cpp
 * @brief Tests for lattice operations — meet/join, dual, LLL, gram, volume, poset.
 */
#include "test_util.hpp"
#include <np/np.hpp>

int main()
{
    using namespace np::lattice;

    // ── Lattice basics: cubic, gram, volume ─────────────────────────────────
    {
        auto L = LatticeFactory::cubic<double>(2);
        test::check(L.rank() == 2 && L.dim() == 2, "cubic rank/dim");
        auto G = L.gram_matrix();
        test::check(std::abs(G(0, 0) - 1) < 1e-9 && std::abs(G(0, 1) - 0) < 1e-9, "cubic gram");
        test::check(std::abs(L.volume() - 1) < 1e-9, "cubic volume");
    }
    {
        auto Lh = LatticeFactory::hexagonal<double>();
        test::check(std::abs(Lh.volume() - std::sqrt(3.0) / 2.0) < 1e-9, "hex volume");
    }
    // ── Dual ─────────────────────────────────────────────────────────────────
    {
        auto L = LatticeFactory::cubic<double>(2);
        auto D = L.dual();
        test::check(std::abs(D.basis(0, 0) - 1) < 1e-9, "dual cubic");
    }
    // ── LLL reduction ────────────────────────────────────────────────────────
    {
        np::lattice::Lattice<double> L({{2, 0}, {1, 1}});
        auto R = L.lll_reduce();
        test::check(R.rank() == 2, "lll rank");
        // shortest vector of Z^2 is 1
        auto Lc = LatticeFactory::cubic<double>(2);
        auto sv = Lc.shortest_vector();
        double nrm = 0;
        for (int i = 0; i < static_cast<int>(sv.size()); ++i)
            nrm += static_cast<double>(sv[i]) * static_cast<double>(sv[i]);
        test::check(std::abs(std::sqrt(nrm) - 1) < 1e-9, "shortest cubic");
    }
    // ── Closest vector (Babai) ───────────────────────────────────────────────
    {
        auto L = LatticeFactory::cubic<double>(2);
        np::ndarray<double> t(std::vector<int>{2});
        t[0] = 0.6;
        t[1] = 0.4;
        auto c = L.closest_vector(t);
        test::check(std::abs(c[0] - 1) < 1e-9 && std::abs(c[1] - 0) < 1e-9, "closest");
    }
    // ── Meet / Join via dual ─────────────────────────────────────────────────
    {
        auto A = LatticeFactory::cubic<double>(2);
        np::ndarray<double> Bm(std::vector<int>{1, 2});
        Bm(0, 0) = 2;
        Bm(0, 1) = 0;
        Lattice<double> B(Bm);
        auto J = A.join(B);
        test::check(J.rank() >= 2, "join rank");
        auto M = A.meet(B);
        // meet of Z^2 and 2Z x Z should be 2Z x Z? rank 2
        test::check(M.rank() == 2 || M.rank() == 1, "meet rank");
    }
    // ── Builder + Strategy + Observer + Visitor ──────────────────────────────
    {
        auto L = LatticeBuilder<double>::create().add_row({2, 0}).add_row({1, 1}).with_lll(0.75).build();
        test::check(L.rank() == 2, "builder lll");
        bool notified = false;
        L.add_observer([&](const Lattice<double> &, const std::string &ev) {
            if (ev == "lll_reduce")
                notified = true;
        });
        auto Lc = L.lll_reduce();
        (void)Lc;
        // observer notified inside lll_reduce via notify
        // we added after, so not notified yet - add before
        Lattice<double> L2({{2, 0}, {1, 1}});
        bool n2 = false;
        L2.add_observer([&](auto &, auto &e) {
            if (e == "lll_reduce")
                n2 = true;
        });
        auto R2 = L2.lll_reduce();
        test::check(n2, "observer lll");
        // Visitor
        struct V : LatticeVisitor<double>
        {
            bool seen = false;
            void visit(const Lattice<double> &) override
            {
                seen = true;
            }
            void visit_basis(const np::ndarray<double> &) override
            {
            }
        } v;
        L.accept(v);
        test::check(v.seen, "visitor lattice");
        // Strategy
        LLLStrategy<double> s;
        auto Rs = L.reduce_with(s);
        test::check(Rs.rank() == 2, "strategy reduce");
        BKZStrategy<double> bkz;
        auto Rb = L.reduce_with(bkz);
        test::check(Rb.rank() == 2, "bkz strategy");
        // Decorator
        np::ndarray<double> Tmat(std::vector<int>{2, 2});
        Tmat(0, 0) = 1;
        Tmat(0, 1) = 0;
        Tmat(1, 0) = 0;
        Tmat(1, 1) = 1;
        TransformedLattice<double> tr(L, Tmat);
        auto Lt = tr.as_lattice();
        test::check(Lt.rank() == 2, "decorator");
        // Free ops (use local lattices)
        {
            auto Af = LatticeFactory::cubic<double>(2);
            np::ndarray<double> Bm2(std::vector<int>{1, 2});
            Bm2(0, 0) = 2;
            Bm2(0, 1) = 0;
            Lattice<double> Bf(Bm2);
            auto fJ = join(Af, Bf);
            auto fM = meet(Af, Bf);
            (void)fJ;
            (void)fM;
            auto g = gram(Af);
            (void)g;
            auto vol = volume(Af);
            test::check(vol > 0, "free vol");
        }
    }
    // ── Poset lattice: boolean, divisor ──────────────────────────────────────
    {
        auto P = LatticeFactory::boolean_lattice(2);
        test::check(P.is_lattice(), "boolean is lattice");
        test::check(P.is_distributive(), "boolean distributive");
        auto m = P.meet(1, 2); // 01 meet 10 = 00
        test::check(m && *m == 0, "boolean meet");
        auto j = P.join(1, 2);
        test::check(j && *j == 3, "boolean join");
        auto hasse = P.hasse_diagram();
        test::check(!hasse.empty(), "hasse");
        auto mu = P.mobius();
        test::check(mu.at({0, 3}) == 1 || mu.at({0, 3}) == -1, "mobius");
        auto zeta = P.zeta();
        test::check(zeta.at({0, 3}) == 1, "zeta");
        // Visitor for poset
        struct PV : PosetVisitor<int>
        {
            bool seen = false;
            void visit(const PosetLattice<int> &) override
            {
                seen = true;
            }
        } pv;
        P.accept(pv);
        test::check(pv.seen, "poset visitor");
    }
    {
        auto D = LatticeFactory::divisor_lattice(12);
        test::check(D.is_lattice(), "divisor is lattice");
        auto m = D.meet(4, 6);
        test::check(m && *m == 2, "divisor meet 4^6=2");
        auto j = D.join(4, 6);
        test::check(j && *j == 12, "divisor join 4v6=12");
    }
    // ── Root lattices A_n, D_n, E8 ───────────────────────────────────────────
    {
        auto A2 = LatticeFactory::a_n<double>(2);
        test::check(A2.rank() == 2 && A2.dim() == 3, "A2");
        auto D4 = LatticeFactory::d_n<double>(4);
        test::check(D4.rank() == 4, "D4");
        auto E8 = LatticeFactory::e8<double>();
        test::check(E8.rank() == 8, "E8");
    }
    // ── Modern C++20: ranges, span, variant, optional ────────────────────────
    {
        Lattice<double> L({{1, 0}, {0, 1}});
        auto sv = L.shortest_vector();
        std::span<double> sp(sv.data().data(), sv.data().size());
        test::check(sp.size() == 2, "span");
        // variant to_variant already tested via Node, here lattice variant via meet/join
        std::variant<Lattice<double>, PosetLattice<int>> var = L;
        test::check(std::holds_alternative<Lattice<double>>(var), "variant");
    }

    return test::failures() ? 1 : 0;
}
