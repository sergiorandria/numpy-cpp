/**
 * @file test_lattice.cpp
 * @brief Tests for lattice operations — meet/join, dual, LLL, gram, volume, poset.
 */
#include "test_util.hpp"
#include <np/np.hpp>

#include <bit>
#include <cmath>
#include <cstdint>
#include <vector>

namespace
{
// Exact determinant of a small integer matrix via doubles: LU with partial
// pivoting is backward stable, and the true determinant is an integer, so
// rounding is exact whenever the accumulated error is below 1/2 (ample
// margin for the well-conditioned matrices tested here; cross-checked by
// mod_det below).
long long lu_det_round(const std::vector<std::vector<long long>> &A)
{
    const int n = static_cast<int>(A.size());
    std::vector<std::vector<double>> M(n, std::vector<double>(n));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            M[i][j] = static_cast<double>(A[i][j]);
    double det = 1.0;
    int sign = 1;
    for (int k = 0; k < n; ++k)
    {
        int piv = k;
        double maxv = std::abs(M[k][k]);
        for (int i = k + 1; i < n; ++i)
        {
            if (std::abs(M[i][k]) > maxv)
            {
                maxv = std::abs(M[i][k]);
                piv = i;
            }
        }
        if (maxv == 0.0)
            return 0;
        if (piv != k)
        {
            M[k].swap(M[piv]);
            sign = -sign;
        }
        det *= M[k][k];
        for (int i = k + 1; i < n; ++i)
        {
            const double f = M[i][k] / M[k][k];
            for (int j = k + 1; j < n; ++j)
                M[i][j] -= f * M[k][j];
        }
    }
    return std::llround(sign * det);
}

long long mod_pow(long long a, long long e, long long p)
{
    long long r = 1;
    a %= p;
    while (e > 0)
    {
        if ((e & 1) != 0)
            r = (r * a) % p;
        a = (a * a) % p;
        e >>= 1;
    }
    return r;
}

// Determinant modulo an odd prime (exact; Gaussian elimination with row
// swaps — a nonzero pivot always exists when p does not divide the true
// determinant, which holds below since det is a power of two).
long long mod_det(const std::vector<std::vector<long long>> &A, long long p)
{
    const int n = static_cast<int>(A.size());
    std::vector<std::vector<long long>> M(n, std::vector<long long>(n));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            M[i][j] = ((A[i][j] % p) + p) % p;
    long long det = 1;
    int sign = 1;
    for (int k = 0; k < n; ++k)
    {
        int piv = -1;
        for (int i = k; i < n; ++i)
        {
            if (M[i][k] != 0)
            {
                piv = i;
                break;
            }
        }
        if (piv < 0)
            return 0;
        if (piv != k)
        {
            M[k].swap(M[piv]);
            sign = -sign;
        }
        det = (det * M[k][k]) % p;
        const long long inv = mod_pow(M[k][k], p - 2, p);
        for (int i = k + 1; i < n; ++i)
        {
            const long long f = (M[i][k] * inv) % p;
            for (int j = k + 1; j < n; ++j)
                M[i][j] = (M[i][j] - f * M[k][j]) % p;
        }
    }
    det = (sign * det) % p;
    return (det + p) % p;
}
// Exact integer-span membership for a full-rank square integer basis.
// Coordinates solve a[j] = Σᵢ A[i][j]·n[i] (rows of A are basis vectors),
// i.e. n = (Aᵀ)⁻¹·a. Doubles locate near-integers (tolerance 1e-6, ample
// margin for the well-conditioned bases here); every acceptance is then
// re-verified exactly in integers, so false positives are impossible and
// false negatives would need >1e-6 solve error.
struct IntSpan
{
    int n = 0;
    std::vector<std::vector<long long>> A;
    std::vector<std::vector<double>> Atinv;
    bool ok = false;

    IntSpan() = default;
    explicit IntSpan(const std::vector<std::vector<long long>> &rows) : n(static_cast<int>(rows.size())), A(rows)
    {
        std::vector<std::vector<double>> M(n, std::vector<double>(2 * n, 0.0));
        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < n; ++j)
                M[i][j] = static_cast<double>(A[j][i]); // transpose: solve in row convention
            M[i][n + i] = 1.0;
        }
        for (int k = 0; k < n; ++k)
        {
            int piv = k;
            double maxv = std::abs(M[k][k]);
            for (int i = k + 1; i < n; ++i)
            {
                if (std::abs(M[i][k]) > maxv)
                {
                    maxv = std::abs(M[i][k]);
                    piv = i;
                }
            }
            if (maxv == 0.0)
                return;
            if (piv != k)
                M[k].swap(M[piv]);
            const double d = M[k][k];
            for (int j = 0; j < 2 * n; ++j)
                M[k][j] /= d;
            for (int i = 0; i < n; ++i)
            {
                if (i == k)
                    continue;
                const double f = M[i][k];
                for (int j = 0; j < 2 * n; ++j)
                    M[i][j] -= f * M[k][j];
            }
        }
        Atinv.assign(n, std::vector<double>(n));
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                Atinv[i][j] = M[i][n + j];
        ok = true;
    }

    bool contains(const long long *a) const
    {
        if (!ok)
            return false;
        std::vector<double> t(n, 0.0);
        for (int i = 0; i < n; ++i)
        {
            double s = 0.0;
            for (int j = 0; j < n; ++j)
                s += Atinv[i][j] * static_cast<double>(a[j]);
            t[i] = s;
        }
        std::vector<long long> nr(n);
        for (int i = 0; i < n; ++i)
        {
            nr[i] = std::llround(t[i]);
            if (std::abs(t[i] - static_cast<double>(nr[i])) > 1e-6)
                return false;
        }
        for (int i = 0; i < n; ++i)
        {
            long long s = 0;
            for (int j = 0; j < n; ++j)
                s += A[j][i] * nr[j];
            if (s != a[i])
                return false;
        }
        return true;
    }
};
} // namespace

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
        WindowedLLLStrategy<double> bkz;
        auto Rb = L.reduce_with(bkz);
        test::check(Rb.rank() == 2, "windowedlll strategy");
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
    // ── E8 determinant (same exact-det machinery as Leech below) ──────────
    {
        auto E8 = LatticeFactory::e8<double>();
        std::vector<std::vector<long long>> A(8, std::vector<long long>(8));
        bool exact = true;
        for (int i = 0; i < 8; ++i)
        {
            for (int j = 0; j < 8; ++j)
            {
                const double v = E8.basis(i, j) * 2.0;
                const long long r = std::llround(v);
                if (v != static_cast<double>(r))
                    exact = false;
                A[i][j] = r;
            }
        }
        test::check(exact, "E8 half-integral basis");
        // det(2B) = ±256 ⟺ det(B) = ±1 (E8 root lattice is unimodular).
        test::check(lu_det_round(A) == 256 || lu_det_round(A) == -256, "E8 det");
        // All 240 roots lie in the span: 112 integer (±1,±1,0⁶) plus 128
        // half-integer ((±½⁸) with even minus signs), in 2B integers.
        {
            const IntSpan span8(A);
            test::check(span8.ok, "E8 span solver ready");
            bool all_in = true;
            for (int i = 0; i < 8 && all_in; ++i)
            {
                for (int j = i + 1; j < 8 && all_in; ++j)
                {
                    for (int si = 0; si < 2 && all_in; ++si)
                    {
                        for (int sj = 0; sj < 2 && all_in; ++sj)
                        {
                            long long a[8] = {};
                            a[i] = si != 0 ? -2 : 2;
                            a[j] = sj != 0 ? -2 : 2;
                            if (!span8.contains(a))
                                all_in = false;
                        }
                    }
                }
            }
            for (int mask = 0; mask < 256 && all_in; ++mask)
            {
                int bits = 0;
                for (int k = 0; k < 8; ++k)
                    bits += (mask >> k) & 1;
                if (bits % 2 != 0)
                    continue;
                long long a[8];
                for (int k = 0; k < 8; ++k)
                    a[k] = ((mask >> k) & 1) != 0 ? -1 : 1;
                if (!span8.contains(a))
                    all_in = false;
            }
            test::check(all_in, "E8 roots in span");
        }
    }
    // ── Leech lattice Λ24 ─────────────────────────────────────────────────
    {
        // Golay code foundation: exact weight enumerator of G24.
        {
            const auto code = detail::golay24_codewords();
            test::check(code.size() == 4096, "golay count");
            int hist[25] = {};
            for (auto w : code)
                hist[std::popcount(w & 0xFFFFFFu)]++;
            test::check(hist[0] == 1 && hist[8] == 759 && hist[12] == 2576 && hist[16] == 759 && hist[24] == 1,
                        "golay enumerator");
        }
        auto L = LatticeFactory::leech<double>();
        test::check(L.rank() == 24 && L.dim() == 24, "leech rank/dim");
        test::check(std::abs(L.volume() - 1.0) < 1e-9, "leech volume");

        // Integer preimage A = round(B·√8); must be exact (entries are k/√8).
        const double s = std::sqrt(8.0);
        std::vector<std::vector<long long>> A(24, std::vector<long long>(24));
        bool exact = true;
        for (int i = 0; i < 24; ++i)
        {
            for (int j = 0; j < 24; ++j)
            {
                const double v = L.basis(i, j) * s;
                const long long r = std::llround(v);
                if (v != static_cast<double>(r))
                    exact = false;
                A[i][j] = r;
            }
        }
        test::check(exact, "leech integral preimage");

        // Determinant ±2^36 (⇒ det Λ24 = 2^-36·2^36 = 1, unimodular):
        // LU-round is exact for this size (error ≪ 1/2), cross-checked by
        // residues mod three odd primes (det has no odd factor to vanish on).
        {
            const long long d = lu_det_round(A);
            test::check(d == 68719476736LL || d == -68719476736LL, "leech det");
            bool residues = true;
            for (long long p : {1000003LL, 1000033LL, 1000037LL})
            {
                long long e = 1;
                long long b = 2 % p;
                long long exp = 36;
                while (exp > 0)
                {
                    if ((exp & 1) != 0)
                        e = (e * b) % p;
                    b = (b * b) % p;
                    exp >>= 1;
                }
                long long got = mod_det(A, p);
                if (got != e && got != (p - e) % p)
                    residues = false;
            }
            test::check(residues, "leech det residues");
        }

        // Evenness + integrality on exact integers: all Gram entries divisible
        // by 8, diagonal even (÷8) — every lattice vector then has even norm.
        {
            bool gram_ok = true;
            for (int i = 0; i < 24 && gram_ok; ++i)
            {
                for (int j = 0; j < 24; ++j)
                {
                    long long g = 0;
                    for (int k = 0; k < 24; ++k)
                        g += A[i][k] * A[j][k];
                    if (g % 8 != 0)
                        gram_ok = false;
                    if (i == j && (g / 8) % 2 != 0)
                        gram_ok = false;
                }
            }
            test::check(gram_ok, "leech even gram");
        }

        // GF(2) row space must be {0, all-ones} (lifts/pairs are even, the odd
        // row is all-odd). This soundly prunes every |a|²=16 pattern except
        // [4×4] (all other patterns have mod-2 weight outside {0, 24}).
        {
            std::uint32_t rs[24];
            for (int i = 0; i < 24; ++i)
            {
                std::uint32_t m = 0;
                for (int j = 0; j < 24; ++j)
                    m |= static_cast<std::uint32_t>((A[i][j] & 1LL) != 0) << j;
                rs[i] = m;
            }
            // Echelon to find the span.
            std::uint32_t basis[24] = {};
            int rank = 0;
            for (int i = 0; i < 24; ++i)
            {
                std::uint32_t v = rs[i];
                for (int k = 0; k < rank; ++k)
                {
                    const unsigned lead = 31u - static_cast<unsigned>(std::countl_zero(basis[k]));
                    if ((v >> lead) & 1u)
                        v ^= basis[k];
                }
                if (v != 0)
                {
                    // Insert keeping descending leading-bit order.
                    const unsigned lead = 31u - static_cast<unsigned>(std::countl_zero(v));
                    int pos = rank;
                    while (pos > 0)
                    {
                        const unsigned pl = 31u - static_cast<unsigned>(std::countl_zero(basis[pos - 1]));
                        if (pl <= lead)
                            break;
                        basis[pos] = basis[pos - 1];
                        --pos;
                    }
                    basis[pos] = v;
                    ++rank;
                }
            }
            test::check(rank == 1 && basis[0] == 0xFFFFFFu, "leech mod-2 row space");
        }

        // Exact span-membership solver shared by the checks below.
        const IntSpan span(A);
        test::check(span.ok, "leech span solver ready");

        // No roots: every |a|²=16 candidate with mod-2 weight 0 (the [4×4]
        // pattern; all other |a|²=16 patterns have weight outside {0, 24} and
        // cannot lie in the span) must fail exact membership.
        {
            long long checked = 0;
            bool found = false;
            for (int i0 = 0; i0 < 24 && !found; ++i0)
                for (int i1 = i0 + 1; i1 < 24 && !found; ++i1)
                    for (int i2 = i1 + 1; i2 < 24 && !found; ++i2)
                        for (int i3 = i2 + 1; i3 < 24 && !found; ++i3)
                        {
                            const int pos[4] = {i0, i1, i2, i3};
                            for (int mask = 0; mask < 16 && !found; ++mask)
                            {
                                long long a[24] = {};
                                for (int k = 0; k < 4; ++k)
                                    a[pos[k]] = ((mask >> k) & 1) != 0 ? -2 : 2;
                                ++checked;
                                if (span.contains(a))
                                    found = true;
                            }
                        }
            test::check(checked == 10626 * 16, "leech root search exhaustive");
            test::check(!found, "leech no roots");
        }

        // Spot checks: all 1104 type-(±4,±4) minimal vectors lie in the span.
        {
            bool all_in = true;
            for (int i = 0; i < 24 && all_in; ++i)
            {
                for (int j = i + 1; j < 24 && all_in; ++j)
                {
                    for (int si = 0; si < 2 && all_in; ++si)
                    {
                        for (int sj = 0; sj < 2 && all_in; ++sj)
                        {
                            long long a[24] = {};
                            a[i] = si != 0 ? -4 : 4;
                            a[j] = sj != 0 ? -4 : 4;
                            if (!span.contains(a))
                                all_in = false;
                        }
                    }
                }
            }
            test::check(all_in, "leech type-1 vectors in span");
        }

        // Spot checks: octad (±2⁸) vectors with even minus signs in span.
        {
            const auto code = detail::golay24_codewords();
            bool all_in = true;
            int sampled = 0;
            const int masks[8] = {0x00, 0x03, 0x05, 0x06, 0x09, 0x0A, 0x0C, 0x0F};
            for (auto w : code)
            {
                if (std::popcount(w & 0xFFFFFFu) != 8)
                    continue;
                int pos[8], np = 0;
                for (int i = 0; i < 24; ++i)
                    if (((w >> i) & 1u) != 0u)
                        pos[np++] = i;
                for (int m = 0; m < 8 && all_in; ++m)
                {
                    long long a[24] = {};
                    for (int k = 0; k < 8; ++k)
                        a[pos[k]] = ((masks[m] >> k) & 1) != 0 ? -2 : 2;
                    if (!span.contains(a))
                        all_in = false;
                    ++sampled;
                }
            }
            test::check(sampled == 759 * 8, "leech type-2 sample count");
            test::check(all_in, "leech type-2 vectors in span");
        }

        // Float instantiation compiles and has full rank.
        {
            auto Lf = LatticeFactory::leech<float>();
            test::check(Lf.rank() == 24 && Lf.dim() == 24, "leech float rank");
        }
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
