/**
 * @file test_fft.cpp
 * @brief Tests for np::fft (radix-2, Bluestein, real-input family,
 *        n-dimensional transforms and frequency helpers).
 */
#include <cmath>
#include <complex>
#include <stdexcept>
#include <vector>

#include "np/np.hpp"
#include "test_util.hpp"

using Cplx = std::complex<double>;
using np::fft::Norm;

namespace
{

bool approx_vec(const std::vector<Cplx> &a, const std::vector<Cplx> &b)
{
    if (a.size() != b.size())
    {
        return false;
    }
    for (std::size_t i = 0; i < a.size(); ++i)
    {
        if (!test::approx_c(a[i], b[i]))
        {
            return false;
        }
    }
    return true;
}

bool approx_flat(const np::ndarray<Cplx> &a, const std::vector<Cplx> &ref)
{
    if (a._numel() != ref.size())
    {
        return false;
    }
    for (std::size_t i = 0; i < ref.size(); ++i)
    {
        if (!test::approx_c(a.data()[i], ref[i]))
        {
            return false;
        }
    }
    return true;
}

bool approx_real_flat(const np::ndarray<double> &a, const std::vector<double> &ref)
{
    if (a._numel() != ref.size())
    {
        return false;
    }
    for (std::size_t i = 0; i < ref.size(); ++i)
    {
        if (!test::approx(a.data()[i], ref[i]))
        {
            return false;
        }
    }
    return true;
}

template <typename T> np::ndarray<T> fill(std::vector<int> shape, std::vector<T> values)
{
    np::ndarray<T> a(shape);
    for (std::size_t i = 0; i < values.size(); ++i)
    {
        a.data()[i] = values[i];
    }
    return a;
}

} // namespace

int main()
{
    // Impulse -> all ones (radix-2, n = 4)
    {
        std::vector<Cplx> x{Cplx{1, 0}, {0, 0}, {0, 0}, {0, 0}};
        auto y = np::fft::fft(x);
        test::check(approx_vec(y, {Cplx{1, 0}, {1, 0}, {1, 0}, {1, 0}}), "fft impulse");
        auto z = np::fft::ifft(y);
        test::check(approx_vec(z, x), "ifft roundtrip n=4");
    }

    // Known 4-point DFT
    {
        std::vector<Cplx> x{Cplx{1, 0}, {2, 0}, {3, 0}, {4, 0}};
        auto y = np::fft::fft(x);
        test::check(test::approx_c(y[0], Cplx{10, 0}), "DFT DC");
        test::check(test::approx_c(y[2], Cplx{-2, 0}), "DFT Nyquist");
        auto z = np::fft::ifft(y);
        test::check(approx_vec(z, x), "roundtrip values");
    }

    // Bluestein path (n = 5, 7 - not powers of two)
    {
        std::vector<Cplx> x5{Cplx{0, 1}, {1, 0}, {2, -1}, {-3, 2}, {0.5, 0.5}};
        auto y = np::fft::fft(x5);
        test::check(y.size() == 5, "bluestein size");
        auto z = np::fft::ifft(y);
        test::check(approx_vec(z, x5), "bluestein roundtrip n=5");

        std::vector<Cplx> x7{Cplx{1, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}};
        auto y7 = np::fft::fft(x7);
        test::check(y7.size() == 7, "bluestein size 7");
        for (const auto &v : y7)
        {
            test::check(test::approx_c(v, Cplx{1, 0}), "bluestein impulse");
        }
    }

    // -----------------------------------------------------------------
    // fft/ifft with axis, n and norm parameters
    // -----------------------------------------------------------------
    {
        // axis = 0 on a 2-D array equals a batched transform of the columns.
        auto a = fill<double>({2, 3}, {1, 2, 3, 4, 5, 6});
        auto y = np::fft::fft(a, std::nullopt, 0);
        // column 0 = {1, 4} -> {5, -3}; column 2 = {3, 6} -> {9, -3}
        test::check(test::approx_c(y(0, 0), Cplx{5, 0}), "fft axis=0 col0 DC");
        test::check(test::approx_c(y(1, 0), Cplx{-3, 0}), "fft axis=0 col0 Nyq");
        test::check(test::approx_c(y(0, 2), Cplx{9, 0}), "fft axis=0 col2 DC");
        auto back = np::fft::ifft(y, std::nullopt, 0);
        test::check(test::approx_c(back(1, 2), Cplx{6, 0}), "ifft axis=0 value");
    }

    {
        // n pads with zeros: fft([1,2,3], 4)
        auto x = fill<double>({3}, {1, 2, 3});
        auto y = np::fft::fft(x, 4);
        test::check(y.shape[0] == 4, "fft pad shape");
        test::check(approx_flat(y, {Cplx{6, 0}, {-2, -2}, {2, 0}, {-2, 2}}), "fft pad values");

        // n crops: fft([1,2,3,4,5], 3) == fft([1,2,3], 3)
        auto b = fill<double>({5}, {1, 2, 3, 4, 5});
        auto y2 = np::fft::fft(b, 3);
        const double s = 0.5 * std::sqrt(3.0);
        test::check(approx_flat(y2, {Cplx{6, 0}, {-1.5, s}, {-1.5, -s}}), "fft truncate");
    }

    // Normalization variants on fft/ifft.
    {
        auto a = fill<double>({4}, {1, 2, 3, 4});
        auto ortho = np::fft::fft(a, std::nullopt, -1, Norm::Ortho);
        test::check(test::approx_c(ortho(0), Cplx{5, 0}), "ortho DC sum/sqrt(n)");
        auto fwd = np::fft::fft(a, std::nullopt, -1, Norm::Forward);
        test::check(test::approx_c(fwd(0), Cplx{2.5, 0}), "forward DC sum/n");

        auto dec = [](const np::ndarray<Cplx> &y, double i) {
            return test::approx_c(y.data()[(std::size_t)i], Cplx{i + 1, 0});
        };
        auto rt_b = np::fft::ifft(np::fft::fft(a), std::nullopt, -1, Norm::Backward);
        auto rt_o = np::fft::ifft(np::fft::fft(a, std::nullopt, -1, Norm::Ortho), std::nullopt, -1, Norm::Ortho);
        auto rt_f = np::fft::ifft(np::fft::fft(a, std::nullopt, -1, Norm::Forward), std::nullopt, -1, Norm::Forward);
        bool ok = true;
        for (int i = 0; i < 4; ++i)
        {
            ok = ok && dec(rt_b, static_cast<double>(i)) && dec(rt_o, static_cast<double>(i)) &&
                 dec(rt_f, static_cast<double>(i));
        }
        test::check(ok, "ifft(fft(x)) == x for all norm modes");
    }

    // -----------------------------------------------------------------
    // rfft / irfft
    // -----------------------------------------------------------------
    {
        auto x = fill<double>({4}, {1, 2, 3, 4});
        auto r = np::fft::rfft(x);
        test::check(r.shape[0] == 3, "rfft length n/2+1");
        test::check(approx_flat(r, {Cplx{10, 0}, {-2, 2}, {-2, 0}}), "rfft values");

        auto back = np::fft::irfft(r);
        test::check(back.shape[0] == 4, "irfft default length 2*(m-1)");
        test::check(approx_real_flat(back, {1, 2, 3, 4}), "irfft roundtrip");
    }

    {
        // Non-power-of-two real transform (Bluestein), odd output length.
        auto a = fill<double>({3}, {2, -1, 0.5});
        auto r = np::fft::rfft(a, 5);
        test::check(r.shape[0] == 3, "rfft n=5 half length");
        auto back = np::fft::irfft(r, 5);
        test::check(back.shape[0] == 5, "irfft odd n");
        test::check(test::approx(back(0), 2) && test::approx(back(1), -1) && test::approx(back(2), 0.5) &&
                        test::approx(back(3), 0) && test::approx(back(4), 0),
                    "irfft odd n roundtrip (zero padded)");

        auto x = fill<double>({7}, {1, 2, 3, 4, 3, 2, 1});
        auto xr = np::fft::rfft(x);
        auto xback = np::fft::irfft(xr, 7);
        bool ok = true;
        for (int i = 0; i < 7; ++i)
            ok = ok && test::approx(xback(i), x(i));
        test::check(ok, "irfft/rfft 7-point (Bluestein) roundtrip");
    }

    {
        // real part discard: rfft(complex) uses only the real part.
        auto a = fill<Cplx>({4}, {Cplx{1, 9}, {2, 8}, {3, 7}, {4, 6}});
        auto r = np::fft::rfft(a);
        test::check(approx_flat(r, {Cplx{10, 0}, {-2, 2}, {-2, 0}}), "rfft discards imaginary part");
    }

    // -----------------------------------------------------------------
    // hfft / ihfft
    // -----------------------------------------------------------------
    {
        // numpy example: hfft([1,2,3,4]) == [15, -4, 0, -1, 0, -4]
        auto a = fill<double>({4}, {1, 2, 3, 4});
        auto h = np::fft::hfft(a);
        test::check(h.shape[0] == 6, "hfft default length 2*(m-1)");
        test::check(approx_real_flat(h, {15, -4, 0, -1, 0, -4}), "hfft example values");
        auto ih = np::fft::ihfft(h);
        test::check(approx_flat(ih, {Cplx{1, 0}, {2, 0}, {3, 0}, {4, 0}}), "ihfft example values");

        // roundtrip for a real signal of even length:
        // ihfft(hfft(x), 2*len(x)-2) == x
        auto x = fill<double>({2}, {5, 6});
        auto hx = np::fft::hfft(x);      // length 2*(m-1) = 2
        auto x2 = np::fft::ihfft(hx, 2); // reconstruct {5, 6}
        test::check(test::approx_c(x2(0), Cplx{5, 0}) && test::approx_c(x2(1), Cplx{6, 0}),
                    "hfft/ihfft 2-point roundtrip");
    }

    // -----------------------------------------------------------------
    // fft2 / ifft2 / fftn / ifftn
    // -----------------------------------------------------------------
    {
        auto a = fill<double>({2, 2}, {1, 2, 3, 4});
        auto y = np::fft::fft2(a);
        test::check(approx_flat(y, {Cplx{10, 0}, {-2, 0}, {-4, 0}, {0, 0}}), "fft2 hand-verified");
        auto back = np::fft::ifft2(y);
        test::check(approx_flat(back, {Cplx{1, 0}, {2, 0}, {3, 0}, {4, 0}}), "ifft2 roundtrip");
    }

    {
        // DC of an all-ones 2x2x2 cube is the volume.
        auto c = fill<double>({2, 2, 2}, {1, 1, 1, 1, 1, 1, 1, 1});
        auto y = np::fft::fftn(c);
        test::check(test::approx_c(y.data()[0], Cplx{8, 0}), "fftn DC");
        auto cy = np::fft::ifftn(y, std::nullopt, std::vector<int>{0, 1, 2});
        bool ok = true;
        for (int i = 0; i < 8; ++i)
            ok = ok && test::approx_c(cy.data()[i], Cplx{1, 0});
        test::check(ok, "ifftn roundtrip");

        // Non-power-of-two along the transformed axes uses Bluestein.
        auto a = fill<double>({3, 5}, {1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1});
        auto y2 = np::fft::fftn(a);
        auto back2 = np::fft::ifftn(y2);
        bool ok2 = true;
        for (int i = 0; i < 15; ++i)
        {
            ok2 = ok2 && test::approx_c(back2.data()[i], Cplx{a.data()[i], 0});
        }
        test::check(ok2, "fftn/ifftn 3x5 Bluestein roundtrip");
    }

    // -----------------------------------------------------------------
    // rfftn / irfftn / rfft2 / irfft2
    // -----------------------------------------------------------------
    {
        auto a = fill<double>({3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
        auto r = np::fft::rfft2(a);
        test::check(r.shape[0] == 3 && r.shape[1] == 2, "rfft2 shape (2 axis half)");
        test::check(test::approx_c(r(0, 0), Cplx{45, 0}), "rfft2 DC");
        auto back = np::fft::irfft2(r, std::vector<int>{3, 3});
        bool ok = true;
        for (int i = 0; i < 9; ++i)
            ok = ok && test::approx(back.data()[i], a.data()[i]);
        test::check(ok, "irfft2 roundtrip");
    }

    // -----------------------------------------------------------------
    // fftfreq / rfftfreq / fftshift / ifftshift
    // -----------------------------------------------------------------
    {
        auto f8 = np::fft::fftfreq(8, 1.0);
        test::check(approx_real_flat(f8, {0, 0.125, 0.25, 0.375, -0.5, -0.375, -0.25, -0.125}), "fftfreq(8,1)");

        auto f7 = np::fft::fftfreq(7, 0.5);
        const double v = 2.0 / 7.0;
        test::check(approx_real_flat(f7, {0, v, 2 * v, 3 * v, -3 * v, -2 * v, -v}), "fftfreq(7,0.5)");

        auto rf8 = np::fft::rfftfreq(8, 1.0);
        test::check(rf8.shape[0] == 5, "rfftfreq length");
        test::check(approx_real_flat(rf8, {0, 0.125, 0.25, 0.375, 0.5}), "rfftfreq(8,1)");
    }

    {
        // 2-D fftshift/ifftshift roundtrip, even and odd sizes.
        auto e = np::fft::fftfreq(8, 0.1);
        auto es = np::fft::fftshift(e);
        auto e_back = np::fft::ifftshift(es);
        bool ok = true;
        for (int i = 0; i < 8; ++i)
            ok = ok && test::approx(e_back(i), e.data()[i]);
        test::check(ok, "ifftshift(fftshift(8))");

        auto o = np::fft::fftfreq(9, 0.1); // odd length
        auto os = np::fft::fftshift(o);
        auto o_back = np::fft::ifftshift(os);
        ok = true;
        for (int i = 0; i < 9; ++i)
            ok = ok && test::approx(o_back.data()[i], o.data()[i]);
        test::check(ok, "ifftshift(fftshift(9)) odd roundtrip");
        // DC now at the center for both parities.
        test::check(test::approx(os.data()[4], 0), "fftshift(9) centered DC");
        test::check(test::approx(es.data()[4], 0), "fftshift(8) centered DC");

        // multi-axis shift on a 2-D frequency grid
        auto g = fill<double>({3, 3}, {0, 1, 2, 3, 4, -4, -3, -2, -1});
        auto gs = np::fft::fftshift(g, std::vector<int>{1});
        // array roll along axis 1 by 3//2 = 1 -> {2,0,1} in the first row
        test::check(test::approx(gs(0, 0), 2) && test::approx(gs(0, 1), 0) && test::approx(gs(0, 2), 1),
                    "fftshift axes=(1) row");
    }

    // -----------------------------------------------------------------
    // Error handling
    // -----------------------------------------------------------------
    {
        bool threw = false;
        np::ndarray<double> empty(std::vector<int>{0});
        try
        {
            (void)np::fft::fft(empty);
        }
        catch (const std::invalid_argument &)
        {
            threw = true;
        }
        test::check(threw, "fft empty axis throws");

        threw = false;
        try
        {
            (void)np::fft::fft(empty, 0);
        }
        catch (const std::invalid_argument &)
        {
            threw = true;
        }
        test::check(threw, "fft n=0 throws");

        auto a2 = fill<double>({2, 2}, {1, 2, 3, 4});
        threw = false;
        try
        {
            (void)np::fft::fft(a2, std::nullopt, 5);
        }
        catch (const np::AxisError &)
        {
            threw = true;
        }
        test::check(threw, "fft axis out of range throws");

        threw = false;
        try
        {
            (void)np::fft::fftn(a2, std::vector<int>{1, 2}, std::vector<int>{0});
        }
        catch (const std::invalid_argument &)
        {
            threw = true;
        }
        test::check(threw, "fftn s/axes mismatch throws");

        threw = false;
        try
        {
            (void)np::fft::fftn(a2, std::nullopt, std::vector<int>{0, 0});
        }
        catch (const std::invalid_argument &)
        {
            threw = true;
        }
        test::check(threw, "fftn duplicate axes throws");

        threw = false;
        try
        {
            auto z = np::fft::irfft(np::fft::rfft(a2));
            (void)z;
        }
        catch (...)
        {
            threw = true;
        }
        test::check(!threw, "rfft/irfft on 2-D input does not throw");
    }

    return test::failures() ? 1 : 0;
}