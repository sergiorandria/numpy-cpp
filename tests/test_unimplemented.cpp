/**
 * @file test_unimplemented.cpp
 * @brief Tests for the formerly-missing np::ndarray members: choose, compress,
 *        partition, dot/matmul methods, real/imag, mT, writeability/base/views,
 *        flat, len, contains, floordiv/divmod/pow, the mod/bitwise/shift
 *        operators, in-place variants and the scalar conversion operators.
 */
#include <complex>
#include <functional>
#include <sstream>
#include <stdexcept>

#include "np/np.hpp"
#include "test_util.hpp"

namespace
{
bool throws_any(std::function<void()> fn)
{
    try
    {
        fn();
    }
    catch (const std::exception &)
    {
        return true;
    }
    return false;
}

bool throws_arg(std::function<void()> fn)
{
    try
    {
        fn();
    }
    catch (const std::invalid_argument &)
    {
        return true;
    }
    catch (...)
    {
    }
    return false;
}
} // namespace

int main()
{
    using np::ndarray;

    // abs / conjugate
    {
        ndarray<int> a = {-3, 0, 4};
        auto ab = a.abs();
        test::check(ab.size() == 3 && ab[0] == 3 && ab[1] == 0 && ab[2] == 4, "abs element-wise");
        ndarray<std::complex<double>> z = {std::complex<double>(3.0, 4.0)};
        test::check(test::approx(static_cast<std::complex<double>>(z.abs()[0]).real(), 5.0), "abs complex");
        ndarray<int> c = {1, -2};
        test::check(c.conjugate()[1] == -2, "conjugate on real type");
        ndarray<std::complex<double>> zc = {std::complex<double>(1.0, 2.0)};
        test::check(test::approx_c(zc.conjugate()[0], std::complex<double>(1.0, -2.0)), "conjugate complex");
    }

    // choose (default 'r', wrap, clip, broadcasting)
    {
        ndarray<int> idx = {1, 0, 2};
        ndarray<int> c0 = {10, 20, 30}, c1 = {40, 50, 60}, c2 = {70, 80, 90};
        auto out = idx.choose(std::vector<ndarray<int>>{c0, c1, c2});
        test::check(out.size() == 3 && out[0] == 40 && out[1] == 20 && out[2] == 90, "choose selects choices[a[i]]");

        ndarray<int> w = {3, -1};
        ndarray<int> w0 = {10, 20}, w1 = {30, 40};
        auto wrap = w.choose(std::vector<ndarray<int>>{w0, w1}, 'w');
        test::check(wrap[0] == 30 && wrap[1] == 40, "choose wrap mode");

        ndarray<int> cl = {5, -1};
        auto clip = cl.choose(std::vector<ndarray<int>>{w0, w1}, 'c');
        test::check(clip[0] == 30 && clip[1] == 20, "choose clip mode");

        test::check(throws_any([&] { cl.choose(std::vector<ndarray<int>>{w0, w1}); }),
                    "choose raise mode throws on out-of-range");

        ndarray<int> sc = {99};
        auto bc = idx.choose(std::vector<ndarray<int>>{sc}, 'c');
        test::check(bc[0] == 99 && bc[1] == 99 && bc[2] == 99, "choose broadcasts choices");
    }

    // compress (axis and flattened)
    {
        ndarray<int> arr = {{1, 2, 3}, {4, 5, 6}};
        ndarray<bool> cond = {true, false, true};
        auto c = arr.compress(cond, 1);
        test::check(c.ndim() == 2 && c.shape[0] == 2 && c.shape[1] == 2, "compress axis 1 shape");
        test::check(c(0, 0) == 1 && c(0, 1) == 3 && c(1, 0) == 4 && c(1, 1) == 6, "compress axis 1 values");

        ndarray<bool> cond0 = {true, false};
        auto c0 = arr.compress(cond0, 0);
        test::check(c0.shape[0] == 1 && c0.shape[1] == 3 && c0(0, 2) == 3, "compress axis 0");

        ndarray<bool> flatc = {true, false, true, true, true, false};
        auto flat = arr.compress(flatc);
        test::check(flat.ndim() == 1 && flat.size() == 4, "compress flattened shape");
        test::check(flat[0] == 1 && flat[1] == 3 && flat[2] == 4 && flat[3] == 5, "compress flattened values");

        test::check(throws_arg([&] { arr.compress(ndarray<bool>{{true, false}}, 1); }),
                    "compress wrong condition length throws");
        test::check(throws_arg([&] { arr.compress(ndarray<bool>{{true, false}, {true, true}}); }),
                    "compress non-1D condition throws");
    }

    // dot / matmul methods delegate to np::linalg
    {
        ndarray<int> a = {{1, 2, 3}, {4, 5, 6}};
        ndarray<int> b = {{7, 8}, {9, 10}, {11, 12}};
        auto d = a.dot(b);
        test::check(d.ndim() == 2 && d.shape[0] == 2 && d.shape[1] == 2, "method dot shape");
        test::check(d(0, 0) == 58 && d(0, 1) == 64 && d(1, 0) == 139 && d(1, 1) == 154, "method dot values");
        auto m = a.matmul(b);
        test::check(m(1, 1) == 154, "method matmul values");
        ndarray<int> v = {1, 2, 3};
        auto vd = a.dot(v);
        test::check(vd.ndim() == 1 && vd.size() == 2 && vd[0] == 14 && vd[1] == 32, "method dot matrix-vector");
    }

    // partition
    {
        ndarray<int> a = {3, 5, 1, 4, 2};
        a.partition(2);
        test::check(a[2] == 3, "partition kth lands in sorted position");
        test::check(a[0] <= 3 && a[1] <= 3 && a[3] >= 3 && a[4] >= 3, "partition separates around kth");

        ndarray<int> b = {{3, 1}, {4, 2}};
        b.partition(0, 1);
        test::check(b(0, 0) == 1 && b(1, 0) == 2, "partition 2D along axis 1");
    }

    // real / imag
    {
        ndarray<std::complex<double>> z = {std::complex<double>(1.0, 2.0), std::complex<double>(-3.0, 4.0)};
        auto re = z.real();
        auto im = z.imag();
        test::check(re.size() == 2 && test::approx(re[0], 1.0) && test::approx(re[1], -3.0), "real of complex");
        test::check(im.size() == 2 && test::approx(im[0], 2.0) && test::approx(im[1], 4.0), "imag of complex");

        ndarray<int> r = {1, 2, 3};
        auto rr = r.real();
        test::check(rr.size() == 3 && rr[2] == 3, "real of real array");
        auto ii = r.imag();
        test::check(ii.size() == 3 && ii[0] == 0 && ii[1] == 0 && ii[2] == 0, "imag of real array is zeros");
    }

    // mT
    {
        ndarray<int> a = {{1, 2, 3}, {4, 5, 6}};
        auto m = a.mT();
        test::check(m.shape[0] == 3 && m.shape[1] == 2, "mT shape");
        test::check(m(0, 1) == 4 && m(2, 0) == 3, "mT values");

        ndarray<int> t(std::vector<int>{2, 3, 4});
        for (int i = 0; i < 24; ++i)
        {
            t.data()[i] = i;
        }
        auto m3 = t.mT();
        test::check(m3.shape[0] == 2 && m3.shape[1] == 4 && m3.shape[2] == 3, "mT 3D shape");
        test::check(m3.get(std::array<std::size_t, 3>{1, 2, 0}) == 14, "mT 3D swaps last two axes");

        test::check(throws_any([&] { ndarray<int>{1, 2}.mT(); }), "mT ndim < 2 throws");
    }

    // setflags / writeable / base / owns_data / is_view
    {
        ndarray<int> a = {1, 2};
        test::check(a.owns_data() && !a.is_view() && a.base() == nullptr, "owned array attributes");
        test::check(a.writeable(), "default writeable");
        a.setflags(false);
        test::check(!a.writeable(), "setflags(false)");

        auto v = a.view();
        test::check(v.is_view() && !v.owns_data() && v.base() != nullptr, "view borrows storage");
        auto c = a.copy();
        test::check(!c.is_view() && c.owns_data(), "copy owns its storage");

        ndarray<int> m = {{1, 2, 3}, {4, 5, 6}};
        test::check(m.is_contiguous() && !m.is_f_contiguous(), "C-order is not F-contiguous");
        auto mt = m.transpose();
        test::check(!mt.is_contiguous() && mt.is_f_contiguous(), "transposed is F-contiguous");
    }

    // flat / len / contains
    {
        ndarray<int> a = {{1, 2, 3}, {4, 5, 6}};
        auto f = a.flat();
        test::check(f.ndim() == 1 && f.size() == 6, "flat is 1-D");
        test::check(f[3] == 4, "flat logical order");
        test::check(a.len() == 2, "len is first axis size");
        test::check(a.contains(5) && !a.contains(9), "contains");
        test::check(throws_arg([&] { (void)ndarray<int>(std::vector<int>{}).len(); }), "len of 0-d throws");
    }

    // floordiv / divmod / pow
    {
        ndarray<int> x = {7, -8, 9};
        auto f = x.floordiv(2);
        test::check(f[0] == 3 && f[1] == -4 && f[2] == 4, "floordiv scalar");
        ndarray<int> d = {2};
        auto fa = x.floordiv(d);
        test::check(fa[0] == 3 && fa[1] == -4, "floordiv array");

        auto r = x % 5;
        test::check(r[0] == 2 && r[1] == 2 && r[2] == 4, "mod numpy semantics");

        auto dm = x.divmod(3);
        test::check(dm.first[0] == 2 && dm.first[1] == -3 && dm.first[2] == 3, "divmod quotient");
        test::check(dm.second[0] == 1 && dm.second[1] == 1 && dm.second[2] == 0, "divmod remainder");

        auto p = x.pow(2);
        test::check(p[0] == 49 && p[1] == 64 && p[2] == 81, "pow");
        auto pf = ndarray<double>{2.5}.pow(2.0);
        test::check(test::approx(pf[0], 6.25), "pow double");
    }

    // operators: %, &, |, ^, ~, <<, >>, unary +/-
    {
        ndarray<int> a = {6, 5};
        ndarray<int> b = {4, 3};
        auto m = a % b;
        test::check(m[0] == 2 && m[1] == 2, "array mod");
        auto ms = a % 3;
        test::check(ms[0] == 0 && ms[1] == 2, "scalar mod");
        auto sl = 10 % a;
        test::check(sl[0] == 4 && sl[1] == 0, "scalar-left mod");

        ndarray<unsigned> u = {5u, 3u};
        auto bw = (u & 1u) | 2u;
        test::check(bw[0] == 3u && bw[1] == 3u, "and/or scalar");
        auto bx = (u & ndarray<unsigned>{3u, 1u}) ^ ndarray<unsigned>{1u, 1u};
        test::check(bx[0] == 0u && bx[1] == 0u, "and/xor array");
        auto nb = ~u;
        test::check(nb[0] == static_cast<unsigned>(~5u) && nb[1] == static_cast<unsigned>(~3u), "bitwise not");

        auto sh = (u << 1u) >> 1u;
        test::check(sh[0] == 5u && sh[1] == 3u, "shifts");
        auto shl = 1u << u;
        test::check(shl[0] == 32u && shl[1] == 8u, "scalar-left shift");

        auto up = +a;
        test::check(up[0] == 6, "unary plus");
        auto un = -a;
        test::check(un[0] == -6 && un[1] == -5, "unary minus");
    }

    // in-place operators
    {
        ndarray<int> a = {7, 8};
        a %= 5;
        test::check(a[0] == 2 && a[1] == 3, "in-place mod");
        a &= 3;
        test::check(a[0] == 2 && a[1] == 3, "in-place and");
        a |= 4;
        test::check(a[0] == 6 && a[1] == 7, "in-place or");
        a ^= 2;
        test::check(a[0] == 4 && a[1] == 5, "in-place xor");
        a <<= 1;
        test::check(a[0] == 8 && a[1] == 10, "in-place left shift");
        a >>= 2;
        test::check(a[0] == 2 && a[1] == 2, "in-place right shift");
        a.floordiv_eq(2);
        test::check(a[0] == 1 && a[1] == 1, "in-place floordiv");
        a.pow_eq(3);
        test::check(a[0] == 1 && a[1] == 1, "in-place pow");

        ndarray<int> b = {5, 9};
        b %= ndarray<int>{2, 4};
        test::check(b[0] == 1 && b[1] == 1, "in-place mod array");
        b &= ndarray<int>{1, 1};
        test::check(b[0] == 1 && b[1] == 1, "in-place and array");
    }

    // scalar conversions
    {
        ndarray<int> one = {42};
        test::check(static_cast<bool>(one) == true, "bool conversion");
        test::check(static_cast<long long>(one) == 42, "long long conversion");
        ndarray<int> zero = {0};
        test::check(static_cast<bool>(zero) == false, "bool conversion zero");
        ndarray<double> fd = {3.5};
        test::check(test::approx(static_cast<double>(fd), 3.5), "double conversion");
        ndarray<std::complex<double>> z = {std::complex<double>(1.0, 2.0)};
        test::check(test::approx_c(static_cast<std::complex<double>>(z), std::complex<double>(1.0, 2.0)),
                    "complex conversion");

        test::check(throws_arg([&] { (void)static_cast<double>(ndarray<int>{1, 2}); }),
                    "scalar conversion of multi-element throws");
    }

    return test::failures() ? 1 : 0;
}
