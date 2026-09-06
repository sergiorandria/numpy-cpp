/**
 * @file test_padic.cpp
 * @brief Tests for p-adic subsystem — Padic, PadicLattice, Hensel, integration.
 */
#include "test_util.hpp"
#include <np/np.hpp>

int main()
{
    using namespace np::padic;

    // ── Padic basics ──────────────────────────────────────────────────────────
    {
        Padic<int64_t> a(5, 7, 10); // 7 mod 5^10
        test::check(a.p == 5 && a.prec == 10, "padic ctor");
        test::check(a.valuation() == 0, "valuation 7 mod5");
        Padic<int64_t> b(5, 25, 10); // 25 = 5^2
        test::check(b.valuation() == 2, "valuation 25");
        test::check(std::abs(b.norm() - 0.04) < 1e-9, "norm 5^-2");
        test::check(a.is_unit(), "is_unit");
        test::check(!b.is_unit(), "not unit");
    }
    {
        Padic<int64_t> a(7, 3, 5);
        Padic<int64_t> b(7, 4, 5);
        auto c = a + b;
        test::check(c.value == 7, "padic add");
        auto d = a * b;
        test::check(d.value == 12, "padic mul");
        auto inv = Padic<int64_t>(5, 3, 5).inverse();
        test::check((Padic<int64_t>(5, 3, 5) * inv).value % 3125 == 1, "padic inv");
    }
    {
        Padic<int64_t> a(5, 7, 10);
        auto exp = a.expansion();
        test::check(exp.size() == 10 && exp[0] == 2, "expansion 7 mod5 -> 2,1");
        // 7 in base 5 is 12 (1*5+2)
        test::check(exp[0] == 2 && exp[1] == 1, "expansion digits");
    }
    // ── Teichmüller ───────────────────────────────────────────────────────────
    {
        Padic<int64_t> a(5, 2, 4); // 2 mod5
        auto t = a.teichmuller();
        test::check(t.is_unit(), "teich unit");
        // t^{p-1} =1 mod p ?
        // 2^4=16 mod5=1, so 2 is unit, teich is 2^{5^{3}}? just check not zero
        test::check(t.value % 5 == 2, "teich residue");
    }
    // ── Factory / Builder / Strategy / Visitor / Observer / Decorator ────────
    {
        auto a = PadicFactory::from_int<int64_t>(5, 7, 10);
        test::check(a.value == 7, "factory from_int");
        auto b = PadicFactory::from_rational<int64_t>(5, 6, 2, 10);
        // 6/2=3 mod 5^10
        test::check(b.value == 3, "factory rational");
        auto z = PadicFactory::zero<int64_t>(7, 5);
        test::check(z.is_zero(), "factory zero");
        auto o = PadicFactory::one<int64_t>(7, 5);
        test::check(o.is_unit(), "factory one");

        auto pb = PadicBuilder<int64_t>::create().prime(7).precision(5).value(3).build();
        test::check(pb.p == 7 && pb.value == 3, "builder");

        // Hensel lift: solve x^2 = 2 mod 7^n, start with x0=3 mod7 (3^2=9=2 mod7)
        Padic<int64_t> x0(7, 3, 6);
        auto f = [](const Padic<int64_t> &x) {
            Padic<int64_t> r(x.p, x.value * x.value - 2, x.prec);
            return r;
        };
        auto df = [](const Padic<int64_t> &x) {
            Padic<int64_t> r(x.p, 2 * x.value, x.prec);
            return r;
        };
        HenselStrategy<int64_t> hs(10);
        auto root = hs.lift(x0, f, df);
        test::check((root.value * root.value - 2) % 7 == 0, "hensel root mod p");
        // Strategy via free function
        NewtonStrategy<int64_t> ns;
        auto root2 = hensel_lift<int64_t>(x0, f, df, ns);
        test::check(root2.value % 7 == 3 || root2.value % 7 == 4, "newton lift");

        // Visitor
        struct V : PadicVisitor<int64_t>
        {
            bool seen = false;
            void visit(const Padic<int64_t> &) override
            {
                seen = true;
            }
        } v;
        a.accept(v);
        test::check(v.seen, "padic visitor");

        // Observer
        bool notified = false;
        Padic<int64_t> obs(5, 7, 5);
        obs.add_observer([&](const Padic<int64_t> &, const std::string &ev) {
            if (ev == "add")
                notified = true;
        });
        auto tmp = obs + Padic<int64_t>(5, 1, 5);
        (void)tmp;
        test::check(notified, "padic observer");

        // Decorator
        ScaledPadic<int64_t> sc(a, 2);
        auto scaled = sc.as_padic();
        test::check(scaled.value == 14, "scaled decorator");

        // Free ops valuation/norm/teich
        test::check(valuation(a) == 0, "free valuation");
        test::check(std::abs(norm(b) - 1) < 1e-9 || norm(b) < 1, "free norm");
        auto tt = teichmuller(a);
        (void)tt;
    }
    // ── PadicLattice ──────────────────────────────────────────────────────────
    {
        auto lat = np::lattice::LatticeFactory::cubic<int64_t>(2);
        PadicLattice<int64_t> pl(lat, 5, 10);
        test::check(pl.rank() == 2, "padic lattice rank");
        auto dual = pl.dual();
        test::check(dual.rank() == 2, "padic dual");
        auto scaled = pl.scaled(1);
        test::check(scaled.rank() == 2, "padic scaled");
        test::check(pl.p_adic_volume() > 0, "p-adic volume");
        test::check(pl.p_adic_norm() > 0, "p-adic norm");
        // meet/join
        auto pl2 = PadicFactory::cubic_padic<int64_t>(2, 5, 10);
        auto j = pl.join(pl2);
        test::check(j.rank() >= 2, "padic join");
        auto m = pl.meet(pl2);
        test::check(m.rank() >= 1, "padic meet");
        // to_padic_lattice integration
        auto pl3 = to_padic_lattice(lat, 5, 10);
        test::check(pl3.rank() == 2, "to_padic_lattice");
        // Padic differential
        PadicDifferential pd(5, 10);
        auto vm_pd = np::differential::VM("x^2", {"x"});
        (void)pd.exterior_derivative(vm_pd);
    }
    // ── Integration with lattice/differential/bigint ──────────────────────────
    {
        // Padic with bigint underlying (use int64 for lattice to avoid LatticeScalar bigint)
        Padic<np::bigint> big(7, np::bigint(123456789), 10);
        test::check(big.valuation() >= 0, "padic bigint");
        // Padic lattice with int (bigint lattice tested via Padic<bigint> above)
        auto lat = np::lattice::Lattice<int64_t>({{1, 0}, {0, 1}});
        PadicLattice<int64_t> plb(lat, 7, 10);
        test::check(plb.rank() == 2, "padic lattice int");
    }
    // ── Modern C++20: span, ranges, variant, optional ────────────────────────
    {
        Padic<int64_t> a(5, 7, 5);
        auto dig = a.expansion();
        std::span<int> sp(dig);
        test::check(sp.size() == 5, "padic span");
        std::variant<Padic<int64_t>, PadicLattice<int64_t>> var = a;
        test::check(std::holds_alternative<Padic<int64_t>>(var), "padic variant");
        std::optional<Padic<int64_t>> opt = a;
        test::check(opt.has_value(), "padic optional");
    }

    return test::failures() ? 1 : 0;
}
