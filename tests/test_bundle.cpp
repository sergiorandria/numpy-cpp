/**
 * @file test_bundle.cpp
 * @brief Tests for bundle.hpp (characteristic classes, Whitney sums, Hodge).
 */
#include "test_util.hpp"
#include <np/np.hpp>

namespace
{
// Minimal stubs: reuse SphereManifold implementations, override only the
// name/dimension dispatch used by characteristic_classes.
// Malformed projective name (no digits after P^).
struct BadNameManifold : np::manifold::SphereManifold
{
    BadNameManifold() : SphereManifold(4)
    {
    }
    std::string name() const override
    {
        return "CP^";
    }
};
// Degenerate stub: negative dimension must not allocate.
struct NegativeDimManifold : np::manifold::SphereManifold
{
    NegativeDimManifold() : SphereManifold(2)
    {
    }
    std::string name() const override
    {
        return "S^-3";
    }
    int dimension() const override
    {
        return -3;
    }
};
} // namespace

int main()
{
    using namespace np;
    using namespace np::bundle;

    // — Exact binomial (old long-long version overflowed at n>=67) —
    {
        auto c = bundle::detail::binom_exact(100, 50);
        test::check(c.convert_to<std::string>() == "100891344545564193334812497256", "binom C(100,50) exact");
        test::check(bundle::detail::binom_exact(10, -1) == 0 && bundle::detail::binom_exact(5, 6) == 0 &&
                        bundle::detail::binom_exact(-4, 2) == 0,
                    "binom out of range");
        test::check(bundle::detail::binom_exact(0, 0) == 1, "binom C(0,0)");
    }
    // — Large CP^n without overflow —
    {
        auto CP70m = manifold::ProjectiveManifold("C", 70);
        auto TCP70 = tangent_bundle(CP70m);
        auto C70 = characteristic_classes(TCP70, &CP70m);
        test::check(C70.chern.size() == 71 && C70.chern[1] == 71, "CP70 chern size/c1");
        test::check(C70.euler == 71, "CP70 euler");
        // C(71,35) = 4537567658043080464 > int64 max would have wrapped
        test::check(C70.chern[35] > 0, "CP70 middle chern positive");
    }
    // — Malformed / degenerate bases: inconclusive, never throw —
    {
        test::check(bundle::detail::parse_projective_n("CP^3") == 3, "parse CP^3");
        test::check(bundle::detail::parse_projective_n("CP^") == -1, "parse empty suffix");
        test::check(bundle::detail::parse_projective_n("CP^3x") == -1, "parse trailing garbage");
        test::check(bundle::detail::parse_projective_n("S^2") == -1, "parse no P^");
        BadNameManifold bad;
        auto Eb = tangent_bundle(bad);
        bool threw = false;
        CharacteristicClasses Cb;
        try
        {
            Cb = characteristic_classes(Eb, &bad);
        }
        catch (...)
        {
            threw = true;
        }
        test::check(!threw && Cb.inconclusive, "malformed name inconclusive");
        NegativeDimManifold neg;
        auto En = tangent_bundle(neg);
        threw = false;
        try
        {
            Cb = characteristic_classes(En, &neg);
        }
        catch (...)
        {
            threw = true;
        }
        test::check(!threw && Cb.inconclusive, "negative dim inconclusive");
        // nullptr base also falls back cleanly
        auto C0 = characteristic_classes(En, nullptr);
        test::check(C0.inconclusive, "null base inconclusive");
    }
    // — Whitney sum: exact sizes and convolution values —
    {
        CharacteristicClasses A, B;
        A.chern = {bigint(1), bigint(2)};
        B.chern = {bigint(1), bigint(3)};
        A.stiefel = {1, 1};
        B.stiefel = {1, 0};
        auto S = whitney_sum_classes(A, B);
        test::check(S.chern.size() == 3, "whitney chern exact size");
        test::check(S.chern[0] == 1 && S.chern[1] == 5 && S.chern[2] == 6, "whitney chern values");
        test::check(S.stiefel.size() == 2 && S.stiefel[0] == 1 && S.stiefel[1] == 1, "whitney stiefel values");
        // Empty inputs treated as identity {1}
        CharacteristicClasses E1, E2;
        auto S0 = whitney_sum_classes(E1, E2);
        test::check(S0.chern.size() == 1 && S0.chern[0] == 1, "whitney empty identity");
        // Bundle-level sum
        auto S2m = manifold::SphereManifold(2);
        auto T2m = manifold::TorusManifold(2);
        auto W = whitney_sum(tangent_bundle(S2m), tangent_bundle(T2m));
        test::check(W.rank == 4 && W.is_orientable, "whitney bundle rank/orient");
    }
    // — Tangent/cotangent/normal —
    {
        auto S2m = manifold::SphereManifold(2);
        auto TS2 = tangent_bundle(S2m);
        auto CT = cotangent_bundle(S2m);
        test::check(CT.name == "T*S^2" && CT.rank == 2, "cotangent");
        auto N = normal_bundle(S2m, 1);
        test::check(N.rank == 0, "normal rank clamped");
        auto Kleinm = manifold::KleinBottleManifold();
        auto CK = characteristic_classes(tangent_bundle(Kleinm), &Kleinm);
        test::check(CK.stiefel.size() == 3 && CK.stiefel[1] == 1 && CK.euler == 0, "klein classes");
        test::check(euler_characteristic_via_euler_class(TS2, &S2m) == 2, "euler via class S2");
        test::check(euler_characteristic_via_euler_class(TS2, nullptr) == 0, "euler via class null");
        // Spheres are stably trivial: total SW w = 1, pontryagin = 1.
        auto CS2 = characteristic_classes(TS2, &S2m);
        test::check(CS2.stiefel.size() == 1 && CS2.stiefel[0] == 1, "sphere SW trivial");
        test::check(CS2.euler == 2, "sphere S2 euler");
        auto S3m = manifold::SphereManifold(3);
        auto CS3 = characteristic_classes(tangent_bundle(S3m), &S3m);
        test::check(CS3.stiefel.size() == 1 && CS3.euler == 0, "sphere S3 trivial/odd euler");
        // Pontryagin of CP^3: p = (1+h^2)^4, so p1 = 4 (catches old C(n+1,2k) bug which gave 6).
        auto CP3m = manifold::ProjectiveManifold("C", 3);
        auto C3 = characteristic_classes(tangent_bundle(CP3m), &CP3m);
        test::check(C3.pontryagin.size() >= 2 && C3.pontryagin[1] == 4, "CP3 pontryagin p1=4");
    }
    // — Hodge: signs, involution, validation —
    {
        HodgeStar hs(2);
        test::check(hs.sign(0) == 1 && hs.sign(1) == -1 && hs.sign(2) == 1, "hodge signs n=2");
        bool threw = false;
        try
        {
            HodgeStar bad(-1);
            (void)bad;
        }
        catch (const std::invalid_argument &)
        {
            threw = true;
        }
        test::check(threw, "hodge negative dim throws");
        // *(dx0) = +dx1 on oriented R^2; *(dx1) = -dx0; ** = -1 on 1-forms.
        differential::KForm dx0(1, 2);
        dx0.coeffs[{0}] = differential::ScalarField([](const differential::Point &) { return 1.0; }, 2);
        auto s1 = hodge_star(dx0, hs);
        test::check(s1.k == 1, "hodge star degree");
        differential::Point p0{0.3, 0.7};
        test::check(std::abs(s1.coeffs[{1}](p0) - 1.0) < 1e-12, "hodge star value");
        differential::KForm dx1(1, 2);
        dx1.coeffs[{1}] = differential::ScalarField([](const differential::Point &) { return 1.0; }, 2);
        auto t1 = hodge_star(dx1, hs);
        test::check(std::abs(t1.coeffs[{0}](p0) + 1.0) < 1e-12, "hodge star dx1=-dx0");
        auto s2 = hodge_star(s1, hs);
        test::check(std::abs(s2.coeffs[{0}](p0) + 1.0) < 1e-12, "hodge involution **=-1");
        threw = false;
        try
        {
            differential::KForm bad(3, 2);
            (void)hodge_star(bad, hs);
        }
        catch (const std::invalid_argument &)
        {
            threw = true;
        }
        test::check(threw, "hodge out-of-range degree throws");
        // n=3: *(dx0) = +dx12 (no (-1)^{k(n-k)} factor on single application).
        {
            HodgeStar hs3(3);
            differential::KForm e0(1, 3);
            e0.coeffs[{0}] = differential::ScalarField([](const differential::Point &) { return 2.0; }, 3);
            auto se0 = hodge_star(e0, hs3);
            differential::Point p3{0.1, 0.2, 0.3};
            test::check(se0.k == 2, "hodge n=3 degree");
            test::check(std::abs(se0.coeffs[{1, 2}](p3) - 2.0) < 1e-12, "hodge n=3 value");
        }
        // Constant forms are harmonic: Laplacian evaluates to ~0.
        differential::KForm f0(0, 2);
        f0.coeffs[{}] = differential::ScalarField([](const differential::Point &) { return 5.0; }, 2);
        auto Lap = laplacian(f0, hs);
        double lv = 0.0;
        for (auto &[idx, field] : Lap.coeffs)
            lv += std::abs(field(p0));
        test::check(lv < 1e-9, "laplacian constant ~0");
        differential::KForm empty;
        test::check(is_harmonic(empty, hs), "empty form harmonic");
        auto delta = codifferential(dx0, hs);
        test::check(delta.k == 0, "codifferential degree");
    }
    return test::failures() ? 1 : 0;
}
